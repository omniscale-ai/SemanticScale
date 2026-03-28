"""Data preparation for SH3: Load QASPER, build 3-level index, extract questions."""

import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.utils import (
    load_config,
    load_jsonl,
    normalize_text,
    save_json,
    save_jsonl,
    sent_tokenize,
)


def load_qasper(split: str) -> list[dict]:
    """Load QASPER dataset from HuggingFace cache.

    Args:
        split: 'train', 'validation', or 'test'

    Returns:
        List of paper dicts with paper_id added.
    """
    logging.info(f"Loading QASPER split: {split}")
    ds = load_dataset("allenai/qasper", split=split)
    papers = []
    for item in ds:
        paper = dict(item)
        papers.append(paper)
    logging.info(f"Loaded {len(papers)} papers from QASPER {split}")
    return papers


def build_paper_index(paper: dict) -> dict:
    """Build 3-level document index for a single paper.

    Returns:
        Dict with keys: paper_id, macro_docs, meso_docs, micro_docs.
        Each doc is {text, doc_id, level, section_index, ...}.
    """
    paper_id = paper["id"]
    section_names = paper["full_text"]["section_name"]
    paragraphs_by_section = paper["full_text"]["paragraphs"]

    macro_docs = []
    meso_docs = []
    micro_docs = []

    # Macro: abstract
    abstract = paper.get("abstract", "")
    if abstract and len(abstract.split()) >= 5:
        macro_docs.append({
            "text": abstract,
            "doc_id": f"{paper_id}__macro__abstract",
            "level": "macro",
            "section_index": -1,
            "section_name": "abstract",
        })

    for sec_idx, (sec_name, sec_paragraphs) in enumerate(
        zip(section_names, paragraphs_by_section)
    ):
        if not sec_paragraphs:
            continue

        # Macro: first paragraph of each section as proxy summary
        first_para = sec_paragraphs[0].strip() if sec_paragraphs[0] else ""
        if first_para and len(first_para.split()) >= 5:
            macro_docs.append({
                "text": first_para,
                "doc_id": f"{paper_id}__macro__s{sec_idx}",
                "level": "macro",
                "section_index": sec_idx,
                "section_name": sec_name or f"section_{sec_idx}",
            })

        for para_idx, para_text in enumerate(sec_paragraphs):
            para_text = para_text.strip()
            if not para_text or len(para_text.split()) < 5:
                continue

            # Meso: full paragraph
            meso_docs.append({
                "text": para_text,
                "doc_id": f"{paper_id}__meso__s{sec_idx}_p{para_idx}",
                "level": "meso",
                "section_index": sec_idx,
                "paragraph_index": para_idx,
                "section_name": sec_name or f"section_{sec_idx}",
            })

            # Micro: sentence chunks
            sentences = sent_tokenize(para_text)
            if not sentences:
                continue

            chunk_size = 3  # sentences per chunk
            overlap = 1

            if len(sentences) <= chunk_size:
                # Whole paragraph is one micro chunk
                chunk_text = " ".join(sentences)
                if len(chunk_text.split()) >= 5:
                    micro_docs.append({
                        "text": chunk_text,
                        "doc_id": f"{paper_id}__micro__s{sec_idx}_p{para_idx}_c0",
                        "level": "micro",
                        "section_index": sec_idx,
                        "paragraph_index": para_idx,
                        "chunk_index": 0,
                        "section_name": sec_name or f"section_{sec_idx}",
                    })
            else:
                step = chunk_size - overlap
                chunk_idx = 0
                for start in range(0, len(sentences), step):
                    end = min(start + chunk_size, len(sentences))
                    chunk_text = " ".join(sentences[start:end])
                    if len(chunk_text.split()) >= 5:
                        micro_docs.append({
                            "text": chunk_text,
                            "doc_id": f"{paper_id}__micro__s{sec_idx}_p{para_idx}_c{chunk_idx}",
                            "level": "micro",
                            "section_index": sec_idx,
                            "paragraph_index": para_idx,
                            "chunk_index": chunk_idx,
                            "section_name": sec_name or f"section_{sec_idx}",
                        })
                        chunk_idx += 1
                    # Stop if we've reached the end
                    if end >= len(sentences):
                        break

    # Include figure/table captions as micro documents
    figures_and_tables = paper.get("figures_and_tables", {})
    captions = figures_and_tables.get("caption", []) if isinstance(figures_and_tables, dict) else []
    for cap_idx, caption in enumerate(captions):
        if caption and isinstance(caption, str) and len(caption.split()) >= 5:
            micro_docs.append({
                "text": caption.strip(),
                "doc_id": f"{paper_id}__micro__caption_c{cap_idx}",
                "level": "micro",
                "section_index": -1,
                "paragraph_index": -1,
                "chunk_index": cap_idx,
                "section_name": "figure_table_caption",
            })

    return {
        "paper_id": paper_id,
        "macro_docs": macro_docs,
        "meso_docs": meso_docs,
        "micro_docs": micro_docs,
    }


def extract_questions(paper: dict) -> list[dict]:
    """Extract questions with gold evidence from one paper.

    Filters out unanswerable questions.

    Returns:
        List of question dicts with: question_text, paper_id, gold_evidence,
        answer_type, question_id.
    """
    paper_id = paper["id"]
    qas = paper.get("qas", {})

    questions_list = qas.get("question", [])
    answers_list = qas.get("answers", [])

    extracted = []

    for q_idx, (question_text, answers_entry) in enumerate(
        zip(questions_list, answers_list)
    ):
        # answers_entry["answer"] is a list of annotator answer dicts
        annotator_answers = answers_entry.get("answer", [])

        # Collect gold evidence and answer types across annotators
        gold_evidence = set()
        answer_types = set()

        for ann_answer in annotator_answers:
            # Determine answer type from QASPER fields:
            #   unanswerable (bool), yes_no (None/bool), extractive_spans (list), free_form_answer (str)
            if ann_answer.get("unanswerable", False):
                answer_types.add("unanswerable")
            elif ann_answer.get("yes_no") is not None:
                answer_types.add("yes_no")
            elif ann_answer.get("extractive_spans") and len(ann_answer["extractive_spans"]) > 0:
                answer_types.add("extractive")
            elif ann_answer.get("free_form_answer"):
                answer_types.add("abstractive")
            else:
                answer_types.add("unknown")

            # Get evidence paragraphs
            evidence = ann_answer.get("evidence", [])
            for ev in evidence:
                if ev and isinstance(ev, str) and ev.strip():
                    gold_evidence.add(ev.strip())

        # Determine primary answer type (most common non-unanswerable)
        non_unanswerable = [t for t in answer_types if t != "unanswerable"]
        if non_unanswerable:
            answer_type = non_unanswerable[0]
        elif answer_types:
            answer_type = "unanswerable"
        else:
            answer_type = "unknown"

        # Filter out unanswerable questions or those with no evidence
        if answer_type == "unanswerable":
            continue
        if not gold_evidence:
            continue

        extracted.append({
            "question_id": f"{paper_id}__q{q_idx}",
            "question_text": question_text,
            "paper_id": paper_id,
            "gold_evidence": sorted(list(gold_evidence)),
            "answer_type": answer_type,
        })

    return extracted


def prepare_all(config: dict, project_root: Path | None = None) -> None:
    """Orchestrate data preparation: load papers, build indices, extract questions.

    Processes both validation and test splits.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    data_dir = project_root / config["paths"]["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing output (checkpoint/resume)
    papers_path = data_dir / "papers_index.jsonl"
    questions_path = data_dir / "questions.jsonl"
    stats_path = data_dir / "index_stats.json"

    if papers_path.exists() and questions_path.exists() and stats_path.exists():
        logging.info("Data preparation output already exists, skipping.")
        return

    all_paper_indices = []
    all_questions = []
    total_macro = 0
    total_meso = 0
    total_micro = 0

    # Process both validation and test splits
    for split in [config["qasper"]["tune_split"], config["qasper"]["eval_split"]]:
        papers = load_qasper(split)

        for paper in tqdm(papers, desc=f"Processing {split}"):
            try:
                # Build 3-level index
                paper_index = build_paper_index(paper)
                paper_index["split"] = split
                all_paper_indices.append(paper_index)

                total_macro += len(paper_index["macro_docs"])
                total_meso += len(paper_index["meso_docs"])
                total_micro += len(paper_index["micro_docs"])

                # Extract questions
                questions = extract_questions(paper)
                for q in questions:
                    q["split"] = split
                all_questions.extend(questions)

            except Exception as e:
                logging.warning(
                    f"Error processing paper {paper.get('id', 'unknown')} "
                    f"in {split}: {e}"
                )
                continue

    # Save outputs
    save_jsonl(all_paper_indices, papers_path)
    save_jsonl(all_questions, questions_path)

    # Compute and save stats
    val_questions = [q for q in all_questions if q["split"] == config["qasper"]["tune_split"]]
    test_questions = [q for q in all_questions if q["split"] == config["qasper"]["eval_split"]]

    stats = {
        "total_papers": len(all_paper_indices),
        "total_macro_docs": total_macro,
        "total_meso_docs": total_meso,
        "total_micro_docs": total_micro,
        "total_questions": len(all_questions),
        "validation_questions": len(val_questions),
        "test_questions": len(test_questions),
        "answer_type_distribution": {},
    }

    # Answer type distribution
    for q in all_questions:
        atype = q["answer_type"]
        stats["answer_type_distribution"][atype] = (
            stats["answer_type_distribution"].get(atype, 0) + 1
        )

    save_json(stats, stats_path)

    logging.info(f"Data preparation complete:")
    logging.info(f"  Papers: {stats['total_papers']}")
    logging.info(f"  Macro docs: {stats['total_macro_docs']}")
    logging.info(f"  Meso docs: {stats['total_meso_docs']}")
    logging.info(f"  Micro docs: {stats['total_micro_docs']}")
    logging.info(f"  Questions: {stats['total_questions']}")
