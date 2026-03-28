# SH1 Rerun: LLM-Labeled Probe

> Probe trained on 2000 LLM-annotated spans (section-blind), tested on 200 LLM-annotated spans.

## Label Agreement (LLM vs SH0 on training set)

- Cohen's κ: 0.3245
- Raw agreement: 0.5534

## Probe Results

| Probe training | F1 vs LLM labels | F1 vs SH0 labels |
|---|---|---|
| **LLM labels** | **0.7543** | 0.6239 |
| SH0 labels | 0.6783 | 0.6845 |
| SH1c residualized | — | 0.4039 |

## Per-Class F1 (LLM-trained probe vs LLM test labels)

              precision    recall  f1-score   support

       macro     0.8649    0.6400    0.7356        50
        meso     0.6528    0.7344    0.6912        64
       micro     0.8132    0.8605    0.8362        86

    accuracy                         0.7650       200
   macro avg     0.7769    0.7449    0.7543       200
weighted avg     0.7748    0.7650    0.7646       200


## Interpretation

The LLM-trained probe achieves F1=0.7543 on LLM test labels, 
compared to the SH0-trained probe's F1=0.6783 on the same labels. 
This measures whether training on content-blind labels improves SLoD detection 
beyond what section-derived labels provide.