# The Case of the Hidden Geometry

## Story Premise

A consulting detective discovers that scientific documents harbour a hidden geometric structure — a continuous axis of abstraction encoded invisibly in the mathematics of language itself — but proving it exists is only the beginning: he must demonstrate it can be controlled, exploited, and observed in the wild, while a string of false leads and inverted clues threaten to discredit the entire investigation.

## Characters

- **Sherlock Holmes** (Lead researcher) — The investigator who first suspects that abstraction levels leave geometric traces in the "embedding space" of documents; relentlessly pursues the five-layer proof
- **Dr. Watson** (Collaborator / narrator) — Documents the journey, runs the parallel field investigations (SH3, SH4, SH5), provides the grounded perspective when Holmes's theories overreach
- **Professor Kanerva** (Remote Finnish GPU laboratory director) — Controls the only apparatus powerful enough to run the critical steering experiments; communicates by telegraph; delivers the decisive result
- **Inspector Lestrade** (The ML research community / conventional methods) — Represents the flat-retrieval, brute-force school; skeptical that geometry matters; proved partially right when hard routing fails
- **Irene Adler** (The SLoD axis itself / the evaluation metric) — The deceptively elegant instrument that both enables and constrains Holmes's conclusions; her "domain" determines what she can and cannot reveal

## Genre & Setting

Victorian London (221B Baker Street) and a remote Finnish laboratory connected by telegraph; classic detective mystery where the crime scene is a corpus of scientific papers and the evidence is mathematical.

## Story Beats

**Beat 1: Setup / The Intrigue Begins**
- **Research events:** SH0 (weak label bootstrap from document structure)
- **Narrative:** Holmes examines fifty scientific manuscripts and notices that titles, methods sections, and footnotes occupy distinct "districts" of abstraction — as reliable as the strata of London clay. He proposes that this structure is not merely organisational but geometrically encoded in the very fabric of language.

**Beat 2: First Clues / The Probe Succeeds**
- **Research events:** SH1 (linear probe, macro-F1 = 0.72, SciBERT embeddings)
- **Narrative:** Holmes constructs a simple instrument — a linear classifier — and demonstrates that a frozen mathematical space already encodes macro, meso, and micro abstraction levels. The reading is clear: F1 = 0.72, with meso the most elusive category, "genuinely ambiguous, like the middle class."
- **Stakes:** The mechanism is confirmed — abstraction geometry is real. But can it be controlled, or is it merely a curiosity?

**Beat 3: Parallel Investigations / Mixed Results**
- **Research events:** SH3 (routing, soft boosting works but hard routing fails), SH4 (drift predicts nothing alone, AUROC = 0.52), SH5 (jump rate null, ρ = 0.003)
- **Narrative:** Watson leads three simultaneous field investigations. Soft retrieval routing yields modest gains — Lestrade grudgingly admits the geometry helps, a little. But drift analysis proves nearly useless. Worst of all, the jump rate hypothesis returns a perfect null: ρ = 0.003. Three leads, one partial, two dead ends.
- **Turning point:** "Holmes's geometry describes how documents are built, not how machines reason. Are we investigating the wrong crime?"

**Beat 4: The Finnish Gambit / Failure After Failure**
- **Research events:** SH2 (doc-span steering d=0.043), SH2-scale, SH2b (|d|=0.546 but inverted), SH2c, SH2a (prompt control ceiling d=0.121)
- **Narrative:** Five experiments arrive back by telegraph from Finland, each a failure. The first: d = 0.043, barely a whisper. The third is the cruelest: magnitude achieved but the direction is inverted. A software bug is found: the layer selection used absolute value instead of signed shift. Five failed experiments. Holmes's reputation is at stake.

**Beat 5: Climax / The Summarisation Breakthrough**
- **Research events:** SH2-summ (summarisation steering, d = 0.679, ROUGE-L improves)
- **Narrative:** Holmes has a revelation. The evaluation instrument was trained on scientific document spans. QA answers are a different genre entirely; she simply cannot read them. But summaries? Summaries *are* document-like text. The sixth experiment: d = 0.679. The steered summaries are closer to the micro reference than the baseline.
- **Revelation:** **Task-domain alignment** — the evaluation axis can only measure what it was trained to see. The five QA failures were measurement failures, not method failures. "The axis was never blind, Watson. We were asking her to read a language she had never learned."

**Beat 6: Resolution / Revisiting the Null**
- **Research events:** SH5a (transition matrix, ρ=−0.197), SH5c (alignment, ρ=−0.135), SH5d (continuous projection, ρ=+0.219)
- **Narrative:** Holmes returns to Watson's "dead" SH5 data. The scalar jump rate was too coarse. Transition matrices reveal two reasoning styles. The final analysis — projecting reasoning traces onto the continuous SLoD axis — yields the strongest result: ρ = +0.219, three times stronger on the SLoD axis than any orthogonal direction.
- **Conclusion:** SLoD is not just a document property, it is a reasoning property.

**Beat 7: The Grand Tournament / The Pairwise Proof**
- **Research events:** SH6 (LLM pairwise SLoD trajectories, AUROC = 0.81)
- **Narrative:** Lestrade remains skeptical: "Embeddings are just shadows. How do we know the model *knows* what level it's at?" Holmes organizes a grand tournament. He asks an LLM judge to compare chunks of reasoning side-by-side. The resulting trajectories are not just elegant; they are predictive. A model's trajectory shape predicts its own success with AUROC = 0.81.
- **Final Note:** The hidden geometry is no longer a shadow; it is a judgment the machines can make themselves.

**Beat 8: Denouement / The Open Case File**
- **Open directions:** Cross-domain portability, absolute SLoD calibration, combined retrieval + steering
- **Final note:** "The geometry is proven, Watson. Now we must learn to speak all its dialects."

## Narrative Arc

Tension builds through a three-act structure: early mechanistic success (Beats 1–2) gives way to mixed and null results (Beat 3), then concentrates into five consecutive failures (Beat 4). The climax pivots on a single conceptual insight — task-domain alignment — that reframes every prior failure as a measurement problem (Beat 5). Resolution comes when "dead" data yields its strongest signal under finer analysis (Beat 6), and the final proof (Beat 7) demonstrates that the SLoD axis is observable even without embeddings, through the eyes of the models themselves.

