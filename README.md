# Model-Based Adaptive Guided Mental-Assessment for Natural Language Depression Screening (MAGMA)

MAGMA is a single-agent, adaptive clinical interview system for the PHQ-8 depression screener. Instead of asking every participant all 8 fixed items, it uses Item Response Theory (IRT); specifically a Graded Response Model (GRM), to estimate a participant's latent depression severity (theta, θ) in real time and adaptively choose the next most informative question to ask.

An LLM (GPT-4o) handles the conversational phrasing of each question and the clinical scoring of each free-text answer. The navigation, meaning which question to ask next and when to stop, is driven entirely by psychometrics rather than the LLM.

## Why adaptive?

The standard PHQ-8 always asks the same 8 questions in the same order. MAGMA instead:
