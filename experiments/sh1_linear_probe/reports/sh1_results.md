# SH1 Results: Linear Decodability of SLoD from Frozen Embeddings

**Generated:** auto

---

## Exit Criteria

- Primary (3-way macro-F1): > 0.6
- Binary fallback (macro-F1): > 0.75
- Baseline gap: > 0.15

## Probe Results

| Model | Classifier | C | Val F1 | Test F1 | Macro P | Macro R |
|-------|-----------|-----|--------|---------|---------|---------|
| minilm | logreg | 0.01 | 0.6536 | 0.6586 | 0.6577 | 0.6618 |
| minilm | svm | 1.0 | 0.6536 | 0.6557 | 0.6564 | 0.6611 |
| scibert | logreg | 0.01 | 0.7044 | 0.7200 | 0.7194 | 0.7217 |
| scibert | svm | 0.01 | 0.7105 | 0.7190 | 0.7189 | 0.7219 |
| specter2 | logreg | 0.01 | 0.7017 | 0.7011 | 0.7004 | 0.7028 |
| specter2 | svm | 0.01 | 0.6998 | 0.6964 | 0.6963 | 0.7008 |

## Baselines

| Baseline | Macro F1 | Accuracy |
|----------|----------|----------|
| random | 0.3283 | 0.3283 |
| majority | 0.1667 | 0.3333 |
| word_count_only | 0.2625 | 0.3294 |
| random_embedding | 0.3348 | 0.3348 |

## Confound Check (full dataset)

- full_dataset_f1: 0.6663034996188814
- full_word_count_f1: 0.3272974065105769
- length_matched_f1: 0.72
- gap: 0.05369650038111862
- note: Based on stratified subsample of 9999 spans (not full 83K)

## Per-Class Analysis (best model)

Best configuration: scibert + logreg (C=0.01)

| Class | F1 |
|-------|-----|
| macro | 0.8206 |
| meso | 0.6156 |
| micro | 0.7237 |

## Verdict

**SH1 CONFIRMED**: Best 3-way test macro-F1 = 0.7200 >= 0.6

Baseline gap: 0.3917 (threshold: 0.15)

## Known Limitations

- QASPER is NLP-papers only; cross-domain transfer deferred to SH2/SH3
- CPU-only embeddings (no GPU fine-tuning)
- Linear probes only (non-linear classifiers deferred)
