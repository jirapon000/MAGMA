# Model-Based Adaptive Guided Mental-Assessment for Natural Language Depression Screening (MAGMA)

MAGMA is a single-agent, adaptive clinical interview system for the PHQ-8 depression screener. Instead of asking every participant all 8 fixed items, it uses Item Response Theory (IRT); specifically a Graded Response Model (GRM), to estimate a participant's latent depression severity (theta, θ) in real time and adaptively choose the next most informative question to ask.

An LLM (GPT-4o) handles the conversational phrasing of each question and the clinical scoring of each free-text answer. The navigation, meaning which question to ask next and when to stop, is driven entirely by psychometrics rather than the LLM.

### Why adaptive?
The standard PHQ-8 always asks the same 8 questions in the same order. MAGMA instead:

1) Estimates a participant's latent trait (θ) from every answer given so far.
2) Ranks the remaining, unasked items by how much new information each would provide at the participant's current θ.
3) Asks the highest-value item next.
4) Stops early once θ is estimated precisely enough (or, in PMI mode, once information gain stalls).
5) Scores every item, including ones never asked — skipped items are predicted from the final θ via the item's fitted Item Characteristic Curve (ICC), so the output is always a full 8-item PHQ-8 profile.

### How it works
1. Parameter estimation (from real data, at startup)
`build_grm_parameters()` fits GRM parameters for each of the 8 PHQ-8 items from the **DAIC-WOZ** ground-truth PHQ-8 labels (Gratch et al., 2014) using girth (marginal maximum likelihood):

- **Discrimination (`a`)** — how sharply an item distinguishes between severity levels.
- **Thresholds (`b1, b2, b3`)** — the θ points at which a participant becomes more likely
  to endorse category `k` over `k-1`.

From the same dataset it also computes:
- A PMI (pointwise mutual information) matrix between item pairs, for the PMI selection strategy.
- Marginal entropy per item, used to weight the entropy-based selection strategy.

> **Note:** DAIC-WOZ is distributed under a data use agreement and is not included in
> this repository. See the [DAIC-WOZ page](https://dcapswoz.ict.usc.edu/) to request access.

2. Theta estimation
Given the ordinal scores (0–3) collected so far, θ is estimated two ways:
- **MAP** (`estimate_theta_map`), using Newton–Raphson optimization of the log-posterior.
- **EAP** (`estimate_theta_eap`), using numerical quadrature over a θ grid (used during the live interview).

A Gaussian prior (`N(0, 1.5)`) regularizes early estimates when little evidence is available.

3. Item selection: three interchangeable strategies
Selectable via `--loss` / `LOSS_FUNCTION`:

| Strategy | Idea | Function |
|---|---|---|
| `fisher` | Picks the item whose ICC is steepest (most discriminating) at the current θ | `fisher_information()` |
| `pmi` | Picks the item most co-associated with symptoms already endorsed | `pmi_gain()` |
| `entropy` | Picks the item expected to most reduce posterior uncertainty in θ, scaled by the item's population variability | `entropy_gain()` |

Stopping rules differ by strategy. Fisher and Entropy stop once the posterior SD of θ is small enough; PMI stops once the best available gain stalls (`PMI_STALL_THRESHOLD`).

4. Conversational question generation
A LangChain prompt (`question_template`) takes the selected clinical domain, the current θ, evidence so far, and conversation history, and asks GPT-4o to phrase one natural, empathetic, single-sentence question. The prompt avoids clinical-sounding phrasing and always weaves in the 2-week PHQ-8 timeframe naturally.

5. Answer scoring: 5-stage Diathesis-Stress Chain-of-Thought (DS-CoT)
Each free-text answer is scored 0–3 by walking GPT-4o through five explicit clinical reasoning stages:

- **Emotion analysis**: affective/somatic state, intensity, polarity, source, trajectory.
- **Contextual grounding**: external stressors, internal framing, stressor magnitude.
- **Attribution resolution**: `PRESENT` / `PRESENT-SITUATIONAL` / `NOT PRESENT`, based on the diathesis-stress model (is the distress proportionate to a named stressor, or does it exceed or predate it?).
- **Reasoning analysis**: contributing or protective factors (social, biological, psychological, functional).
- **Calibrated severity estimation**: final 0–3 PHQ-8 score, judged on functional impact rather than on how much or how emotionally the participant spoke.

This extends the 4-stage CoT scoring approach of Teng et al. (2025) with an added contextual-grounding stage and three-way (rather than binary) symptom attribution.

6. A screening pre-question
Before any PHQ-8 item, a broad wellbeing question is asked and scored the same way. Its score initializes θ (via the `PHQ_8Depressed` item as a proxy) so that even the first real PHQ-8 item is chosen adaptively, rather than starting from a flat prior.

7. Final batch scoring
At the end of the interview, all 8 items are scored, regardless of whether they were asked:

- **Asked items** use the LLM-derived score from the interview.
- **Unasked items** are predicted from the final θ via the item's ICC, `P(X=k | θ)`, with the MAP category taken as the score.

Each item also gets a natural-language clinical explanation (LLM-generated) and a data-sufficiency rating (`HIGH` / `MEDIUM` / `LOW`) based on how peaked its ICC probability distribution is.



