# SH0 -- Weak Label Bootstrap: Design

## Hypothesis

A heuristically labeled dataset of text spans tagged as macro/meso/micro using document
structure can achieve sufficient quality (>75% agreement with human/LLM judges) to train
an SH1 linear probe with >0.60 macro-F1.

## Method

### Data Source

QASPER (`allenai/qasper`): 1,585 NLP papers with structured `full_text` including explicit
section names and paragraph-level granularity.

### Heuristic Labeling Rules

Labeling operates at the sentence level; paragraph labels are assigned by majority vote.

**Step 1: Section-Name Classification**
- Section names split on `:::` delimiters (QASPER subsection hierarchy)
- Each segment classified via `re.search` with `\b` word boundaries
- MACRO: abstract, introduction, conclusion, summary, related work, background, overview, motivation
- MESO: approach, method(s), model, framework, dataset(s), evaluation, analysis, discussion, limitations
- MICRO: experiment(s), result(s), implementation details, hyperparameters, ablation, appendix, metrics, benchmark
- Most specific match wins (micro > meso > macro)

**Step 2: Position Rules**
- First 2 paragraphs of Introduction -> macro
- Last paragraph of Introduction (if >2 paras) -> meso
- Any paragraph in Conclusion/Summary -> macro
- First paragraph of unrecognized section -> meso

**Step 3: Content-Based Micro Score Override**
Weighted sum of signals: decimal numbers (0.25), table refs (0.15), figure refs (0.15),
equation refs (0.10), entity density (0.15), citation density (0.10), metric values (0.10).
If micro_score > 0.35, sentence labeled micro regardless of section/position.

**Step 4: Fallback**
Unrecognized sections with no content signal default to meso.

**Step 5: Confidence Scoring**
- 0.9: section regex matched and paragraph label agrees
- 0.6: section regex matched but disagrees
- 0.3: section name unrecognized

### Output Format

Paragraph-level JSONL with fields: span_id, paper_id, text, label, label_source,
confidence, section_name, micro_score, word_count, sentence_count.

Two outputs:
1. Full dataset (`qasper_slod_spans.jsonl`)
2. Length-matched subset (`qasper_slod_length_matched.jsonl`) -- equal spans per class per
   word-count bucket, critical for controlling length confound in SH1.

## Analysis / Exit Criteria

| Criterion | Threshold |
|---|---|
| Total labeled spans | >10,000 paragraph-level |
| Label balance | No class <15% of total |
| LLM/manual agreement | >75% on 100-sample stratified check |
| Length-matched subset | >2,000 spans (>660 per class) |
| Automated checks | All 6 pass (distribution, word count, section coverage, abstract=macro, conclusion=macro, micro in experiments sections) |

## Results

All exit criteria met after 4 iterations:
- 83,135 labeled spans (macro=21.8%, meso=62.8%, micro=15.3%)
- 84.9% manual agreement on 86 stratified samples
- 37,278 length-matched spans (12,426 per class)
- 6/6 automated checks passed
