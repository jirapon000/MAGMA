# Model-Based Adaptive Guided Mental-Assessment for Natural Language Depression Screening (MAGMA)

MAGMA is a single-agent, adaptive clinical interview system for the PHQ-8 depression screener. Instead of asking every participant all 8 fixed items, it uses Item Response Theory (IRT); specifically a Graded Response Model (GRM), to estimate a participant's latent depression severity (theta, θ) in real time and adaptively choose the next most informative question to ask.

An LLM (GPT-4o) handles the conversational phrasing of each question and the clinical scoring of each free-text answer. The navigation, meaning which question to ask next and when to stop, is driven entirely by psychometrics rather than the LLM.

## Why adaptive?
The standard PHQ-8 always asks the same 8 questions in the same order. MAGMA instead:

1) Estimates a participant's latent trait (θ) from every answer given so far.
2) Ranks the remaining, unasked items by how much new information each would provide at the participant's current θ.
3) Asks the highest-value item next.
4) Stops early once θ is estimated precisely enough (or, in PMI mode, once information gain stalls).
5) Scores every item, including ones never asked — skipped items are predicted from the final θ via the item's fitted Item Characteristic Curve (ICC), so the output is always a full 8-item PHQ-8 profile.

## How it works
