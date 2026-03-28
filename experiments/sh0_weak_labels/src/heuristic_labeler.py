"""Step 2: Apply heuristic labeling rules to raw QASPER data.

Labeling strategy (from PLAN.md):
  1. Section-name regex -> macro/meso/micro
  2. Position rules within section
  3. Content-based micro_score
  4. Combined label = majority vote of sentence-level labels

Output schema per span:
  span_id, paper_id, text, label, label_source, confidence,
  section_name, section_index, paragraph_index, micro_score,
  char_count, word_count, sentence_count

Usage:
    python src/heuristic_labeler.py [--input data/qasper_raw.jsonl] [--output data/qasper_slod_spans.jsonl]
"""

import argparse
import re
from collections import Counter

import spacy
from tqdm import tqdm

from src.utils import (
    load_config,
    resolve_path,
    read_jsonl,
    write_jsonl,
    count_words,
    count_chars,
    setup_logging,
)

logger = setup_logging("heuristic_labeler")

# ---------------------------------------------------------------------------
# Section-name regex patterns (PLAN.md 2.1)
# ---------------------------------------------------------------------------

MACRO_SECTIONS = re.compile(
    r"(?i)\b(abstract|introduction|conclusions?|summary|"
    r"related\s*works?|background|overview|motivation|"
    r"acknowledge?ments?)\b"
)

MESO_SECTIONS = re.compile(
    r"(?i)\b(approach|methods?|methodology|models?|framework|system|"
    r"proposed|architecture|design|setup|formulation|"
    r"problem\s*(statement|definition|formulation)|"
    r"task\s*(description|definition)|"
    r"datasets?|evaluation|analysis|"
    r"discussion|limitations?|future\s*work|"
    r"baselines?|features?|encoder|decoder|"
    r"embeddings?|training|learning|preprocessing|annotation|"
    r"classification|representation|corpus|corpora|"
    r"data\s*(collection|quality|processing))\b"
)

MICRO_SECTIONS = re.compile(
    r"(?i)\b(experiments?|experimental|"
    r"results?|implementation(\s+details)?|"
    r"hyperparameters?|ablation|"
    r"appendix|supplementary|"
    r"metrics?|performance|benchmark|"
    r"examples?|case\s*stud|settings?|supplemental|"
    r"comparisons?|statistics?|visualization|"
    r"qualitative|empirical|test)\b"
)


def _classify_single_segment(name: str) -> str | None:
    """Classify a single section-name segment (no delimiter handling)."""
    name = name.strip()
    if not name:
        return None
    if MACRO_SECTIONS.search(name):
        return "macro"
    if MESO_SECTIONS.search(name):
        return "meso"
    if MICRO_SECTIONS.search(name):
        return "micro"
    return None


# Priority order: most specific wins
_LEVEL_PRIORITY = {"micro": 0, "meso": 1, "macro": 2}


def classify_section_name(section_name: str) -> str | None:
    """Classify a section name into macro/meso/micro or None if unknown.

    Handles ``:::``-delimited section names (e.g. ``"Methods ::: Experiments"``)
    by classifying each segment independently and returning the most specific
    match (micro > meso > macro).
    """
    if not section_name:
        return None
    segments = [s for s in re.split(r'\s*:::\s*', section_name) if s]
    best = None
    for seg in segments:
        level = _classify_single_segment(seg)
        if level is not None:
            if best is None or _LEVEL_PRIORITY[level] < _LEVEL_PRIORITY[best]:
                best = level
    return best


# ---------------------------------------------------------------------------
# Content-based micro score (PLAN.md 2.3)
# ---------------------------------------------------------------------------

def compute_micro_score_from_doc(doc, sentence: str) -> float:
    """Score 0-1 indicating how 'micro' a sentence is.

    Accepts a pre-processed spaCy Doc (or span) to avoid redundant nlp() calls.
    """
    signals = {
        "has_numbers": bool(re.search(r"\d+\.\d+|\d+%", sentence)),
        "has_table_ref": bool(re.search(r"(?i)(table|tab\.)\s*\d", sentence)),
        "has_figure_ref": bool(re.search(r"(?i)(figure|fig\.)\s*\d", sentence)),
        "has_equation_ref": bool(re.search(r"(?i)(equation|eq\.)\s*\d", sentence)),
        "entity_density": len(list(doc.ents)) / max(len(doc), 1),
        "citation_density": (
            len(re.findall(r"\([^)]*\d{4}[^)]*\)", sentence))
            / max(len(sentence.split()), 1)
        ),
        "has_specific_value": bool(
            re.search(
                r"(?:F1|accuracy|BLEU|ROUGE|precision|recall)\s*(?:of|=|:)?\s*\d",
                sentence,
                re.I,
            )
        ),
    }

    score = (
        0.25 * signals["has_numbers"]
        + 0.15 * signals["has_table_ref"]
        + 0.15 * signals["has_figure_ref"]
        + 0.10 * signals["has_equation_ref"]
        + 0.15 * min(signals["entity_density"] * 5, 1.0)
        + 0.10 * min(signals["citation_density"] * 10, 1.0)
        + 0.10 * signals["has_specific_value"]
    )
    return round(score, 4)


def compute_micro_score(sentence: str, nlp) -> float:
    """Score 0-1 indicating how 'micro' a sentence is (legacy wrapper)."""
    doc = nlp(sentence)
    return compute_micro_score_from_doc(doc, sentence)


# ---------------------------------------------------------------------------
# Sentence-level labeling
# ---------------------------------------------------------------------------

def label_sentence(
    sentence: str,
    nlp,
    section_level: str | None,
    section_name: str,
    paragraph_index: int,
    total_paragraphs: int,
    micro_threshold: float,
    intro_macro_paragraphs: int,
    section_lead_is_meso: bool,
    *,
    sent_doc=None,
) -> tuple[str, str, float]:
    """Label a single sentence. Returns (label, label_source, micro_score)."""
    if sent_doc is not None:
        micro_score = compute_micro_score_from_doc(sent_doc, sentence)
    else:
        micro_score = compute_micro_score(sentence, nlp)
    _sname = (section_name or "").strip()
    is_introduction = bool(re.match(r"(?i)^introduction$", _sname))
    is_conclusion = bool(re.match(r"(?i)^(conclusion|summary)$", _sname))

    # Start with section-level label
    label = section_level
    source = "section_regex" if section_level else "unknown"

    # Position rules (PLAN.md 2.2)
    # First 2 paragraphs of Introduction -> macro
    if is_introduction and paragraph_index < intro_macro_paragraphs:
        label = "macro"
        source = "position"

    # Last paragraph of Introduction -> meso (contribution list)
    if is_introduction and paragraph_index == total_paragraphs - 1 and total_paragraphs > intro_macro_paragraphs:
        label = "meso"
        source = "position"

    # Conclusion paragraphs -> macro
    if is_conclusion:
        label = "macro"
        source = "position" if not section_level else source

    # First paragraph of any section -> meso (section lead)
    if section_lead_is_meso and paragraph_index == 0 and label is None:
        label = "meso"
        source = "position"

    # Content override: if micro_score exceeds threshold, override to micro
    if micro_score > micro_threshold:
        label = "micro"
        source = "content" if source in ("unknown", None) else "combined"

    # Fallback: if still no label, use content signals or default to meso
    if label is None:
        if micro_score > 0.3:
            label = "micro"
            source = "content"
        else:
            label = "meso"
            source = "position"

    return label, source, micro_score


# ---------------------------------------------------------------------------
# Paragraph-level labeling (majority vote of sentence labels)
# ---------------------------------------------------------------------------

def label_paragraph(
    text: str,
    nlp,
    section_level: str | None,
    section_name: str,
    paragraph_index: int,
    total_paragraphs: int,
    config_heuristics: dict,
) -> tuple[str, str, float, float, int]:
    """Label a paragraph via majority vote of its sentence labels.

    Returns: (label, label_source, confidence, micro_score_avg, sentence_count)
    """
    micro_threshold = config_heuristics["micro_score_threshold"]
    intro_macro_paras = config_heuristics["intro_macro_paragraphs"]
    section_lead = config_heuristics["section_lead_is_meso"]
    conf = config_heuristics["confidence"]

    doc = nlp(text)
    sent_spans = [sent for sent in doc.sents if sent.text.strip()]

    if not sent_spans:
        return "meso", "position", conf["low"], 0.0, 0

    sentences = [sent.text.strip() for sent in sent_spans]
    sentence_labels = []
    sentence_sources = []
    micro_scores = []

    for sent_span, sent_text in zip(sent_spans, sentences):
        lbl, src, ms = label_sentence(
            sent_text, nlp, section_level, section_name,
            paragraph_index, total_paragraphs,
            micro_threshold, intro_macro_paras, section_lead,
            sent_doc=sent_span,
        )
        sentence_labels.append(lbl)
        sentence_sources.append(src)
        micro_scores.append(ms)

    # Majority vote for paragraph label
    label_counts = Counter(sentence_labels)
    label = label_counts.most_common(1)[0][0]

    # Determine label source
    source_counts = Counter(sentence_sources)
    primary_source = source_counts.most_common(1)[0][0]
    # If multiple sources contributed, call it "combined"
    if len(source_counts) > 1:
        primary_source = "combined"

    # Confidence scoring
    avg_micro = sum(micro_scores) / len(micro_scores)
    if section_level is not None:
        # Section regex matched
        if label == section_level:
            # Content agrees with section label
            confidence = conf["high"]
        else:
            # Content disagrees
            confidence = conf["medium"]
    else:
        # Section name unrecognized
        confidence = conf["low"]

    return label, primary_source, confidence, round(avg_micro, 4), len(sentences)


# ---------------------------------------------------------------------------
# Main labeling pipeline
# ---------------------------------------------------------------------------

def label_all_spans(raw_records: list[dict], config: dict, nlp) -> list[dict]:
    """Label all paragraphs from raw QASPER records.

    Returns list of span dicts matching the output schema from PLAN.md 3.1.
    Uses nlp.pipe() for efficient batch processing.
    """
    heuristics = config["heuristics"]

    # Phase 1: Collect all paragraphs that need NLP processing, and
    # immediately handle Title records (no NLP needed).
    title_spans = []
    nlp_jobs = []  # list of (text, metadata_dict)

    logger.info("Phase 1: Collecting paragraphs...")
    for record in raw_records:
        paper_id = record["paper_id"]
        section_name = record["section_name"]
        section_index = record["section_index"]
        paragraphs = record["paragraphs"]

        if section_name == "Title":
            for pi, para in enumerate(paragraphs):
                span_id = f"{paper_id}_s{section_index}_p{pi}"
                title_spans.append({
                    "span_id": span_id,
                    "paper_id": paper_id,
                    "text": para,
                    "label": "macro",
                    "label_source": "section_regex",
                    "confidence": heuristics["confidence"]["high"],
                    "section_name": section_name,
                    "section_index": section_index,
                    "paragraph_index": pi,
                    "micro_score": 0.0,
                    "char_count": count_chars(para),
                    "word_count": count_words(para),
                    "sentence_count": 1,
                })
            continue

        section_level = classify_section_name(section_name) if section_name != "Abstract" else "macro"
        total_paras = len(paragraphs)

        for pi, para in enumerate(paragraphs):
            if not para.strip():
                continue
            nlp_jobs.append((para, {
                "paper_id": paper_id,
                "section_name": section_name,
                "section_index": section_index,
                "paragraph_index": pi,
                "section_level": section_level,
                "total_paras": total_paras,
                "is_abstract": section_name == "Abstract",
            }))

    logger.info(f"Phase 1: {len(title_spans)} title spans, {len(nlp_jobs)} paragraphs to process")

    # Phase 2: Batch NLP processing
    logger.info("Phase 2: Batch NLP processing...")
    texts = [job[0] for job in nlp_jobs]
    metas = [job[1] for job in nlp_jobs]

    micro_threshold = heuristics["micro_score_threshold"]
    intro_macro_paras = heuristics["intro_macro_paragraphs"]
    section_lead = heuristics["section_lead_is_meso"]
    conf = heuristics["confidence"]

    nlp_spans = []
    for doc, meta in tqdm(
        zip(nlp.pipe(texts, batch_size=256), metas),
        total=len(texts),
        desc="Labeling paragraphs",
    ):
        paper_id = meta["paper_id"]
        section_name = meta["section_name"]
        section_index = meta["section_index"]
        pi = meta["paragraph_index"]
        section_level = meta["section_level"]
        total_paras = meta["total_paras"]
        is_abstract = meta["is_abstract"]
        para = doc.text

        sent_spans_list = [s for s in doc.sents if s.text.strip()]
        sents = [s.text.strip() for s in sent_spans_list]

        if is_abstract:
            avg_ms = 0.0
            if sents:
                avg_ms = sum(
                    compute_micro_score_from_doc(sp, sp.text.strip())
                    for sp in sent_spans_list
                ) / len(sents)
            span_id = f"{paper_id}_s{section_index}_p{pi}"
            nlp_spans.append({
                "span_id": span_id,
                "paper_id": paper_id,
                "text": para,
                "label": "macro",
                "label_source": "section_regex",
                "confidence": conf["high"],
                "section_name": section_name,
                "section_index": section_index,
                "paragraph_index": pi,
                "micro_score": round(avg_ms, 4),
                "char_count": count_chars(para),
                "word_count": count_words(para),
                "sentence_count": len(sents),
            })
            continue

        # Regular body paragraph: label via sentence majority vote
        if not sents:
            span_id = f"{paper_id}_s{section_index}_p{pi}"
            nlp_spans.append({
                "span_id": span_id,
                "paper_id": paper_id,
                "text": para,
                "label": "meso",
                "label_source": "position",
                "confidence": conf["low"],
                "section_name": section_name,
                "section_index": section_index,
                "paragraph_index": pi,
                "micro_score": 0.0,
                "char_count": count_chars(para),
                "word_count": count_words(para),
                "sentence_count": 0,
            })
            continue

        sentence_labels = []
        sentence_sources = []
        micro_scores = []

        for sent_span, sent_text in zip(sent_spans_list, sents):
            lbl, src, ms = label_sentence(
                sent_text, nlp, section_level, section_name,
                pi, total_paras,
                micro_threshold, intro_macro_paras, section_lead,
                sent_doc=sent_span,
            )
            sentence_labels.append(lbl)
            sentence_sources.append(src)
            micro_scores.append(ms)

        label_counts = Counter(sentence_labels)
        label = label_counts.most_common(1)[0][0]

        source_counts = Counter(sentence_sources)
        primary_source = source_counts.most_common(1)[0][0]
        if len(source_counts) > 1:
            primary_source = "combined"

        avg_micro = sum(micro_scores) / len(micro_scores)
        if section_level is not None:
            if label == section_level:
                confidence = conf["high"]
            else:
                confidence = conf["medium"]
        else:
            confidence = conf["low"]

        span_id = f"{paper_id}_s{section_index}_p{pi}"
        nlp_spans.append({
            "span_id": span_id,
            "paper_id": paper_id,
            "text": para,
            "label": label,
            "label_source": primary_source,
            "confidence": confidence,
            "section_name": section_name,
            "section_index": section_index,
            "paragraph_index": pi,
            "micro_score": round(avg_micro, 4),
            "char_count": count_chars(para),
            "word_count": count_words(para),
            "sentence_count": len(sents),
        })

    # Combine title spans + nlp-processed spans
    spans = title_spans + nlp_spans
    return spans


def print_distribution(spans: list[dict]) -> None:
    """Print label distribution summary."""
    label_counts = Counter(s["label"] for s in spans)
    total = len(spans)
    logger.info(f"Total spans: {total}")
    for label in ["macro", "meso", "micro"]:
        count = label_counts.get(label, 0)
        pct = count / total * 100 if total > 0 else 0
        logger.info(f"  {label}: {count} ({pct:.1f}%)")

    source_counts = Counter(s["label_source"] for s in spans)
    logger.info("Label sources:")
    for src, count in source_counts.most_common():
        logger.info(f"  {src}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Apply heuristic labeling to QASPER data")
    parser.add_argument("--input", type=str, default=None, help="Input JSONL path")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = resolve_path(args.input or config["paths"]["raw_data"])
    output_path = resolve_path(args.output or config["paths"]["labeled_spans"])

    logger.info(f"Loading raw data from {input_path}")
    raw_records = read_jsonl(input_path)
    logger.info(f"Loaded {len(raw_records)} section records")

    logger.info("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "attribute_ruler"])

    logger.info("Labeling spans...")
    spans = label_all_spans(raw_records, config, nlp)

    write_jsonl(spans, output_path)
    logger.info(f"Wrote {len(spans)} spans to {output_path}")

    print_distribution(spans)


if __name__ == "__main__":
    main()
