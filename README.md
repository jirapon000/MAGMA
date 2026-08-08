# Model-Based Adaptive Guided Mental-Assessment for Natural Language Depression Screening (MAGMA)

MAGMA is a single-agent, adaptive clinical interview system for the PHQ-8 depression screener. Instead of asking every participant all 8 fixed items, it uses Item Response Theory (IRT); specifically a Graded Response Model (GRM), to estimate a participant's latent depression severity (theta, θ) in real time and adaptively choose the next most informative question to ask.

An LLM (GPT-4o) handles the conversational phrasing of each question and the clinical scoring of each free-text answer. The navigation, meaning which question to ask next and when to stop, is driven entirely by psychometrics rather than the LLM.

---

### Why adaptive?
The standard PHQ-8 always asks the same 8 questions in the same order. MAGMA instead:

1) Estimates a participant's latent trait (θ) from every answer given so far.
2) Ranks the remaining, unasked items by how much new information each would provide at the participant's current θ.
3) Asks the highest-value item next.
4) Stops early once θ is estimated precisely enough (or, in PMI mode, once information gain stalls).
5) Scores every item, including ones never asked — skipped items are predicted from the final θ via the item's fitted Item Characteristic Curve (ICC), so the output is always a full 8-item PHQ-8 profile.
   
---

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

Stopping rules differ by strategy. Fisher and Entropy stop once the posterior SD of θ drops to 0.3 or below; PMI stops once the best available gain improves by less than 0.10 (`PMI_STALL_THRESHOLD`).

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

---

### Usage
MAGMA can be run two ways: as a terminal CLI interview, or as a local web demo.

#### Option 1: CLI (terminal interview)

```bash
python MAGMA.py --id P001 --loss fisher     # or: pmi | entropy
```

| Flag | Description |
|---|---|
| `--id` | Session identifier used to label output files (e.g. `P001`) |
| `--loss` | Item-selection strategy: `fisher` (default), `pmi`, or `entropy` |

The interview runs directly in the terminal: MAGMA prints each question, and you type your answer at the `💬 Participant:` prompt.

#### Option 2: Web demo (Flask + browser UI)

```bash
python app.py
```

Then open **http://127.0.0.1:8000** in your browser. Choose a strategy (Fisher / PMI / Entropy) on the landing screen and click **Start Simulation** to begin chatting with the agent.

- No `--id` or `--loss` CLI flags are used here; the loss function is chosen via the UI and sent to the server per session.
- Each browser session is tracked in memory with a randomly generated session ID (no output files are written to disk automatically like the CLI version).

#### Requirements (both options)

- Python 3.9+ (adjust if you know your actual tested version)
- `OPENAI_API_KEY` set in a `.env` file (loaded via `python-dotenv`)
- A ground-truth PHQ-8 dataset at `Dataset/PHQ8 Mapping/GrouthTruth_PHQ8_Labels.csv`, containing the 8 PHQ-8 columns:
  `PHQ_8NoInterest, PHQ_8Depressed, PHQ_8Sleep, PHQ_8Tired, PHQ_8Appetite, PHQ_8Failure, PHQ_8Concentrating, PHQ_8Moving`
- Python packages:
```bash
  pip install numpy pandas torch openai girth python-dotenv langchain-openai langchain-core langgraph flask
```
  
---

### Architecture
MAGMA has one shared "brain" (`MAGMA.py`) with all the psychometric logic (GRM, θ estimation, item selection, DS-CoT scoring). It can be run two different ways:

#### 1. CLI version (`MAGMA.py`)

Runs as a step-by-step flow using LangGraph:

```
Ask question → Get answer → Update theta → Pick next question → repeat → Final scoring
```

You answer questions directly in the terminal, and results are saved to files at the end.

#### 2. Web demo version (`app.py` + `index.html`)

Same brain, different wrapper. Instead of LangGraph, it uses a simple Flask web server:

- You open a browser and chat with the agent through a chat UI.
- Each time you send an answer, the server scores it, updates theta, and sends back the next question (or the final results).

The logic is the same either way, just delivered through the terminal vs. a browser.

---
### Dataset

MAGMA involves two distinct datasets:
### 1. Training data

Used to fit the GRM item parameters at startup. See [How it works → Parameter estimation](#1-parameter-estimation-from-real-data-at-startup) for details on DAIC-WOZ and how fitting works.

### 2. Participant experiment data (collected via MAGMA)

Running MAGMA with real participants produces a dataset containing:

| Field | Description |
|---|---|
| `Gender`, `Age`, `Edu_Level`, `Employ_Status`, `Marital`, `Country`, `Primary_Lang` | Participant demographics collected alongside the interview |
| `Mental_Health_History`, `Diagnosis` | Self-reported mental health background |
| PHQ-8 item scores (0–3 per item, total 0–24) | Ground-truth score for each item, reviewed and validated by a licensed psychologist |
| Free-text answers | The participant's actual sentence-form responses to each question, as recorded in the transcript |

This dataset is used as a validation set: comparing MAGMA's automated PHQ-8 scores against the psychologist-verified ground-truth scores for the same participants.

> Note: this dataset is separate from the DAIC-WOZ training data mentioned above and is not currently included in this repository. [Add a note on access/privacy/IRB status if applicable.]

---

### Output

**Note:** File output only happens in the CLI version (`MAGMA.py`). The web demo (`app.py`) keeps everything in memory and does not save files.

Each CLI run creates a folder `MAGMA/<loss_function>/` containing:

| Folder | Contents |
|---|---|
| `Evidence/` | Per-item supporting/contradicting/neutral evidence extracted during the interview |
| `Transcript/` | Full turn-by-turn transcript (`.jsonl`) with θ at each turn |
| `Agent_Thoughts/` | Navigation log: θ, posterior SD, items asked, decision at each turn |
| `Scores/` | Final PHQ-8 scores per item, total score, severity category, θ, items asked, loss function used |
| `Scoring_Explanations/` | ICC probabilities, entropy, sufficiency, and LLM-generated clinical explanation per item |
| `Analysis_Metrics/` | Per-turn analytics: DS-CoT stage outputs, θ, posterior SD, PMI gains, agent vs. parsed scores |
| `Symptoms/` | Per-item θ-at-time-of-asking summary |
| `GRM_Gains/` | Per-turn ranking of candidate items and their information-gain scores |

Files are named using `--id`, e.g. `Scores_P001.csv`, `Transcript_P001.jsonl`.

---

#### Severity bands (total PHQ-8 score, 0–24)

The PHQ-8 total score is the sum of all 8 item scores (each scored 0–3), producing a range from 0 to 24. This total is mapped to a severity category using the standard PHQ-8 classification scheme (Kroenke et al., 2009), the same convention used in psychological research and clinical practice to classify a person's depression severity based on their score:

| Score | Category |
|---|---|
| 0 | No Depression |
| 1–4 | Minimal Depression |
| 5–9 | Mild Depression |
| 10–14 | Moderate Depression |
| 15–19 | Moderately Severe Depression |
| 20–24 | Severe Depression |

A higher total score reflects greater self-reported depressive symptom burden over the past two weeks. Note that this categorization is a screening heuristic, not a clinical diagnosis; it's intended to flag the likely severity of symptoms and guide follow-up, not to replace evaluation by a mental health professional.


