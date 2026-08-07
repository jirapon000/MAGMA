# ============================ MAGMA ========================================
#  Single adaptive agent conducting a PHQ-8 clinical interview.
#  Navigation uses a Graded Response Model (GRM) to estimate latent depression
#  severity (theta) from real participant responses.
#  The Question Agent uses LLM to ask the 8 PHQ-8 items and adaptive follow-ups.
#  Three interchangeable information-gain modules select the next best question:
#    1. Fisher Information  — maximise information at current theta estimate
#    2. PMI                 — pointwise mutual information from ground-truth data
#    3. Entropy             — expected reduction in posterior entropy
#  Theta and all IRT parameters (a, b_k) are estimated from the real dataset.
#  No simulated client — designed for live human participants.
# ==============================================================================
#  python MAGMA.py --id P001 --loss fisher (pmi or entropy)
#
#  FLOW (per turn)
#  ─────────────────────────────────────────────────────────────────
#  STARTUP
#    build_grm_parameters()  → item discrimination + threshold params
#    build_pmi_matrix()       → PMI matrix (for PMI mode)
#
#  PER TURN (adaptive agent loop)
#    Question Agent  → GRM-ranked candidate → LLM phrases question
#    Human answers   → score_response_cosine()   → ordinal score 0-3 (cosine similarity)
#    update_theta()  → MAP/EAP update of latent trait theta
#    select_next_item()  → chosen by LOSS_FUNCTION setting
#    Navigation      → NEXT_ITEM (pure GRM — always moves forward)
#
#  END
#    batch_scoring_node → scoring from LLM if that question being asked and if not asked use theta to match with ICC threshold (b1,b2,b3)
# ==============================================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import json
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import openai
import re
import girth

from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END

load_dotenv()

# =============================================================================
#  CONFIG
# =============================================================================
DATASET_PATH        = "Dataset/PHQ8 Mapping/GrouthTruth_PHQ8_Labels.csv"
AI_NAME             = "Adaptive Assessment Agent"
PARTICIPANT_NAME    = "Participant"
LLM_MODEL           = "gpt-4o"
LLM_TEMPERATURE     = 0.7
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")

# ── Information-gain module: choose ONE of "fisher", "pmi", "entropy"
LOSS_FUNCTION       = "fisher"   # options: "fisher" | "pmi" | "entropy"
PMI_STALL_THRESHOLD    = 0.10   # stop PMI if best gain increases by less than this

# ── GRM / theta estimation
THETA_PRIOR_MEAN    = 0.0
THETA_PRIOR_SD      = 1.5
THETA_GRID_POINTS   = 200        # resolution for numerical EAP integration

# ── NLI thresholds (keep from MAGMA)

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please add it to your .env file.")

# =============================================================================
#  PHQ-8 ITEM DEFINITIONS
# =============================================================================
ITEMS = [
    "PHQ_8NoInterest", "PHQ_8Depressed", "PHQ_8Sleep", "PHQ_8Tired",
    "PHQ_8Appetite",   "PHQ_8Failure",   "PHQ_8Concentrating", "PHQ_8Moving"
]

PHQ8_HYPOTHESES = [
    {"item_id": "I1", "label": "Anhedonia",       "phq_key": "PHQ_8NoInterest",    "text": "I have lost interest or pleasure in activities I used to enjoy."},
    {"item_id": "I2", "label": "Depressed mood",  "phq_key": "PHQ_8Depressed",     "text": "I feel down, depressed, or hopeless."},
    {"item_id": "I3", "label": "Sleep problems",  "phq_key": "PHQ_8Sleep",         "text": "I have trouble sleeping or I sleep too much."},
    {"item_id": "I4", "label": "Fatigue",         "phq_key": "PHQ_8Tired",         "text": "I feel tired or have little energy."},
    {"item_id": "I5", "label": "Appetite change", "phq_key": "PHQ_8Appetite",      "text": "I have a poor appetite or I am overeating."},
    {"item_id": "I6", "label": "Self-worth",      "phq_key": "PHQ_8Failure",       "text": "I feel bad about myself or that I have let my family down."},
    {"item_id": "I7", "label": "Concentration",   "phq_key": "PHQ_8Concentrating", "text": "I have trouble concentrating on things."},
    {"item_id": "I8", "label": "Psychomotor",     "phq_key": "PHQ_8Moving",        "text": "I have been moving or speaking slowly, or feeling fidgety and restless."},
]

ITEM_ID_TO_PHQ_KEY = {h["item_id"]: h["phq_key"] for h in PHQ8_HYPOTHESES}
PHQ_KEY_TO_ITEM    = {h["phq_key"]: h for h in PHQ8_HYPOTHESES}
ITEM_ID_TO_ITEM    = {h["item_id"]: h for h in PHQ8_HYPOTHESES}

PHQ8_CLINICAL_CONTEXT = {
    "PHQ_8NoInterest":    "little interest or pleasure in doing things",
    "PHQ_8Depressed":     "feeling down, depressed, or hopeless",
    "PHQ_8Sleep":         "trouble falling/staying asleep, or sleeping too much",
    "PHQ_8Tired":         "feeling physically or mentally drained, having little energy or motivation to do things, needing more rest than usual, or feeling exhausted even after sleeping",
    "PHQ_8Appetite":      "poor appetite or overeating",
    "PHQ_8Failure":       "feeling like a failure or that you've let people down",
    "PHQ_8Concentrating": "trouble concentrating on things",
    "PHQ_8Moving":        "moving or speaking slower than usual, or feeling restless/fidgety",
}


# =============================================================================
#  SCREENING QUESTION
#  Asked once before PHQ-8 items to get an initial theta estimate.
#  Answer is scored on the same 0-3 rubric as PHQ-8 items using a broad
#  depression proxy hypothesis, then fed into estimate_theta_map() so the
#  first real PHQ-8 item is chosen adaptively for each individual.
# =============================================================================
SCREENING_QUESTION = (
    "Before we get into some more specific questions, I'd love to get a general sense "
    "of how you've been feeling overall. Over the last two weeks, how would you describe "
    "your general mood and energy — have things been feeling pretty okay, or has it been "
    "more of a struggle lately?"
)
SCREENING_HYPOTHESIS = (
    "Over the last two weeks I have been feeling down, low in energy, or struggling "
    "with my mood on most days."
)
SCREENING_LABEL = "General Wellbeing Screening"


# =============================================================================
#  STEP 1 — GRM PARAMETER ESTIMATION FROM DATASET
# Graded Response Model (Samejima 1969):
#   P*(theta, k) = 1 / (1 + exp(-a * (theta - b_k)))
#   P(X=k|theta) = P*(k) - P*(k+1)   where P*(0)=1, P*(K)=0
#
# Parameters estimated via marginal MLE approximation:
#   a_j   = discrimination for item j  (estimated from item-total correlation)
#   b_k_j = threshold for category k of item j
#             estimated as: logit(P(X >= k)) / a_j
#
# Also computes PMI matrix and item entropy from marginal distributions.
# =============================================================================
def build_grm_parameters(dataset_path: str): #use by FIsher, Entropy and ICC Scoring
    """
    Estimate GRM item parameters (discrimination a, thresholds b_k) from
    the real PHQ-8 ground-truth dataset.  Returns:
      grm_params : dict  {phq_key: {"a": float, "b": [b1, b2, b3]}}
      pmi_matrix : pd.DataFrame  (for PMI loss function)
      item_entropy: dict  {phq_key: float}  (marginal entropy of each item)
    """
    df = pd.read_csv(dataset_path)
    df_scores = df[ITEMS].clip(0, 3).astype(int)
    n_items   = len(ITEMS)

    # ── Fit GRM using Maximum Marginal Likelihood — 
    responses_matrix = df_scores[ITEMS].values.astype(int).T  # shape: (8, N)
    estimates        = girth.grm_mml(responses_matrix)
    a_params         = estimates['Discrimination']   # shape: (8,)
    difficulty_matrix = estimates['Difficulty']      # shape: (8, 3)

    grm_params = {
        ITEMS[i]: {
            "a": round(float(a_params[i]), 4),
            "b": [round(float(difficulty_matrix[i, k]), 4) for k in range(3)]
        }
        for i in range(len(ITEMS))
    }

    # ── PMI Matrix (binary co-occurrence)
    df_bin  = (df_scores >= 2).astype(int) #Score 0 and 1 = absent and Score 2 and 3 = present 
    pmi_matrix = pd.DataFrame(np.zeros((n_items, n_items)), index=ITEMS, columns=ITEMS)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            p_a  = df_bin[ITEMS[i]].mean()  # P(x)   — marginal probability of item i
            p_b  = df_bin[ITEMS[j]].mean()  # P(y)   — marginal probability of item j
            p_ab = ((df_bin[ITEMS[i]] == 1) & (df_bin[ITEMS[j]] == 1)).mean() # P(x,y) — joint probability both present
            if p_ab > 0 and p_a > 0 and p_b > 0:
                pmi = np.log2(p_ab / (p_a * p_b)) # PMI(x,y) = log2(P(x,y) / P(x)×P(y))
                val = max(0.0, pmi)
                pmi_matrix.iloc[i, j] = pmi_matrix.iloc[j, i] = val

    # ── Item marginal entropy H(X_j) — used for entropy-based selection
    item_entropy = {}
    for col in ITEMS:
        counts = df_scores[col].value_counts(normalize=True)
        h      = -sum(p * np.log(p + 1e-12) for p in counts if p > 0)
        item_entropy[col] = round(h, 6)


    print("GRM parameters ready.")
    print(f"  Discrimination range: {min(a_params):.2f} – {max(a_params):.2f}")
    print("PMI matrix ready.")
    # return grm_params, pmi_matrix, item_entropy
    # DEBUG: sanity check GRM parameter signs
    print("\n🔍 DEBUG grm_params:")
    for key, p in grm_params.items():
        print(f"  {key}: a={p['a']:.3f}, b={p['b']}")

    return grm_params, pmi_matrix, item_entropy
# =============================================================================
#  STEP 2 — GRM PROBABILITY FUNCTIONS
#  GRM Mathematical Formula
# =============================================================================
def grm_p_star(theta: float, a: float, b_k: float) -> float:
    """P*(X >= k | theta) = logistic(a*(theta - b_k))"""
    return 1.0 / (1.0 + np.exp(-a * (theta - b_k)))

def grm_category_probs(theta: float, a: float, b_list: list) -> np.ndarray:
    """
    Returns [P(X=0), P(X=1), P(X=2), P(X=3)] given GRM params.
    Uses boundary convention P*(X>=0)=1, P*(X>=4)=0.
    """
    boundaries = [1.0] + [grm_p_star(theta, a, bk) for bk in b_list] + [0.0]
    probs      = np.diff(-np.array(boundaries))   # P(X=k) = P*(k) - P*(k+1)
    probs      = np.clip(probs, 1e-8, 1.0)
    probs     /= probs.sum()                        # normalise for safety
    return probs


# =============================================================================
#  STEP 3 — THETA ESTIMATION (MAP)
# Empirical Bayes MAP update using Newton–Raphson on log-posterior.
# =============================================================================
def compute_log_posterior(theta: float, responses: Dict[str, int],
                           grm_params: Dict) -> float:
    """log P(theta | responses) ∝ log P(responses | theta) + log prior"""
    # Prior: N(0, 1)
    log_prior = -0.5 * ((theta - THETA_PRIOR_MEAN) / THETA_PRIOR_SD) ** 2
    log_lik   = 0.0
    for phq_key, score in responses.items():
        if phq_key not in grm_params:
            continue
        a      = grm_params[phq_key]["a"]
        b_list = grm_params[phq_key]["b"]
        probs  = grm_category_probs(theta, a, b_list)
        k      = min(score, len(probs) - 1)
        log_lik += np.log(probs[k] + 1e-12)
    return log_prior + log_lik
# MAximum A Posteriori (MAP)
#"Given everything the participant has said so far, what is the most likely depression severity level on the -4 to +4 scale?"
def estimate_theta_map(responses: Dict[str, int], grm_params: Dict,
                        init_theta: float = 0.0) -> float:
    if not responses:
        return THETA_PRIOR_MEAN
    theta = init_theta
    for _ in range(50):
        h  = 1e-4
        f0 = compute_log_posterior(theta,     responses, grm_params)
        f1 = compute_log_posterior(theta + h, responses, grm_params)
        f2 = compute_log_posterior(theta - h, responses, grm_params)
        grad = (f1 - f2) / (2 * h)
        hess = (f1 - 2 * f0 + f2) / (h ** 2)
        if abs(hess) < 1e-10:
            break
        step = grad / hess
        theta -= step
        theta  = np.clip(theta, -4.0, 4.0)
        if abs(step) < 1e-6:
            break
    return round(float(theta), 5)

def estimate_theta_eap(responses: Dict[str, int], grm_params: Dict) -> float:
    """EAP (expected a posteriori) via numerical quadrature on a grid."""
    grid   = np.linspace(-4.0, 4.0, THETA_GRID_POINTS)
    log_w  = np.array([
        compute_log_posterior(t, responses, grm_params)
        for t in grid
    ])
    log_w -= log_w.max()          # numerical stability
    w      = np.exp(log_w)
    w     /= w.sum()
    return round(float(np.dot(w, grid)), 5)


# =============================================================================
#  STEP 4 — LOSE FUNCTIONS
# ── FISHER INFORMATION at theta for item j
#    I_j(theta) = sum_k  (dP_k/dtheta)^2 / P_k
# ── PMI gain: sum of PMI(j, s) * evidence(s)  for already-answered s
# ── ENTROPY gain: expected posterior entropy reduction if we observe item j
# =============================================================================
def fisher_information(theta: float, phq_key: str, grm_params: Dict) -> float:
    """Fisher information for item j at current theta estimate."""
    if phq_key not in grm_params:
        return 0.0
    a      = grm_params[phq_key]["a"]
    b_list = grm_params[phq_key]["b"]
    h      = 1e-5
    probs0 = grm_category_probs(theta - h, a, b_list)
    probs1 = grm_category_probs(theta + h, a, b_list)
    dp     = (probs1 - probs0) / (2 * h)
    probs  = grm_category_probs(theta, a, b_list)
    fi     = np.sum((dp ** 2) / (probs + 1e-12))
    return float(fi)

def pmi_gain(phq_key: str, pmi_matrix: pd.DataFrame,
             item_responses: Dict[str, int]) -> float:
    """
    PMI-weighted information gain — sum of PMI connections to answered items.
    Uses floor weight so score 0 still contributes a small signal.
    """
    total = 0.0
    for s, score in item_responses.items():
        if s != phq_key and s in pmi_matrix.columns:
            weight = max(score / 3.0, 0.1)
            total += weight * pmi_matrix.loc[phq_key, s]
    return float(total)

def entropy_gain(theta: float, phq_key: str,
                 responses: Dict[str, int], grm_params: Dict,
                 item_entropy: Dict[str, float] = None) -> float:
    """
    Expected entropy reduction (information gain) if we observe item j.

    Two entropy sources are combined:
      1. Posterior theta entropy reduction — how much knowing item j narrows
         the estimate of the latent trait theta.
         ΔH_theta = H(theta | past responses) - E_k[H(theta | past + X_j=k)]
         Computed numerically via EAP grid.

      2. Marginal item entropy H(X_j) — pre-computed from the real dataset.
         Items that are more variable in the population (high marginal entropy)
         carry more potential information.
         Stored in item_entropy[phq_key] from build_grm_parameters().

    Final gain = ΔH_theta × (1 + H(X_j) / log(4))
      — the dataset-derived marginal entropy scales up items that are
        inherently more discriminating across the real participant population.
    """
    if phq_key not in grm_params:
        return 0.0
    a      = grm_params[phq_key]["a"]
    b_list = grm_params[phq_key]["b"]

    # ── Part 1: posterior theta entropy reduction
    grid    = np.linspace(-4.0, 4.0, THETA_GRID_POINTS)
    log_w0  = np.array([compute_log_posterior(t, responses, grm_params) for t in grid])
    log_w0 -= log_w0.max()
    w0      = np.exp(log_w0); w0 /= w0.sum()
    h_prior = -np.sum(w0 * np.log(w0 + 1e-12))

    expected_h_post = 0.0
    for k in range(4):
        p_k = float(np.sum(w0 * np.array([
            grm_category_probs(t, a, b_list)[k] for t in grid
        ])))
        if p_k < 1e-8:
            continue
        new_resp  = dict(responses)
        new_resp[phq_key] = k
        log_wk  = np.array([compute_log_posterior(t, new_resp, grm_params) for t in grid])
        log_wk -= log_wk.max()
        wk      = np.exp(log_wk); wk /= wk.sum()
        h_post  = -np.sum(wk * np.log(wk + 1e-12))
        expected_h_post += p_k * h_post

    delta_h_theta = float(max(0.0, h_prior - expected_h_post))

    # ── Part 2: marginal item entropy from CSV (pre-computed)
    # H(X_j) = -Σ P(X=k) log P(X=k) over the real population
    # Normalised by log(4) (max entropy for 4 categories) → range [0, 1]
    if item_entropy and phq_key in item_entropy:
        h_item_norm = item_entropy[phq_key] / np.log(4)   # normalise to [0,1]
    else:
        h_item_norm = 0.5   # neutral fallback if not provided

    # Combined gain: posterior reduction scaled by population variability
    combined_gain = delta_h_theta * (1.0 + h_item_norm)
    return float(combined_gain)

def select_next_item(
    theta: float,
    asked_keys: List[str],
    responses: Dict[str, int],
    grm_params: Dict,
    pmi_matrix: pd.DataFrame,
    item_entropy: Dict[str, float] = None,
    loss_fn: str = "fisher",
) -> List[str]:
    """
    Rank unasked items by the chosen information-gain criterion.
    Returns sorted list of phq_keys (best first).

    Loss functions:
      fisher  — Fisher information I_j(theta) at current theta estimate.
                Picks the item whose ICC is steepest at the current trait level.
                Uses GRM params pre-computed from CSV.

      pmi     — PMI-weighted gain: Σ evidence(s) × PMI(j, s) / evidence(j).
                Picks the item most co-associated with already-observed symptoms.
                PMI matrix pre-computed from CSV co-occurrence counts.

      entropy — Expected posterior theta entropy reduction × population variability.
                delta_H_theta scaled by marginal H(X_j) pre-computed from CSV.
                Most principled but slowest (numerical integration).
    """
    remaining = [k for k in ITEMS if k not in asked_keys]
    if not remaining:
        return []
    scores = {}
    for phq_key in remaining:
        if loss_fn == "fisher":
            scores[phq_key] = fisher_information(theta, phq_key, grm_params)
        elif loss_fn == "pmi":
            if not responses:
                scores[phq_key] = sum(
                    pmi_matrix.loc[phq_key, s]
                    for s in pmi_matrix.columns
                    if s != phq_key
        )
            else:
                scores[phq_key] = pmi_gain(phq_key, pmi_matrix, responses)
        elif loss_fn == "entropy":
            scores[phq_key] = entropy_gain(
                theta, phq_key, responses, grm_params, item_entropy
            )
        else:
            raise ValueError(f"Unknown LOSS_FUNCTION: {loss_fn!r}. Choose fisher | pmi | entropy")
    return sorted(remaining, key=scores.get, reverse=True)


# =============================================================================
#  STEP 5 — LLM SCORER
# Converts a free-text participant answer to an ordinal PHQ-8 score (0-3)
# using LLM (GPT-4o). More accurate than cosine similarity for understanding
# negation, context, and nuanced clinical language.
# =============================================================================
def score_with_llm(answer_text: str, item_label: str,
                   item_hypothesis: str, llm: ChatOpenAI) -> tuple[int, dict]:
    """
    Five-stage Diathesis-Stress Chain-of-Thought (DS-CoT) scoring.
    Extends the 4-stage CoT of Teng et al. (2025) by inserting Contextual
    Grounding after emotion analysis and replacing binary classification
    with three-way attribution (PRESENT / PRESENT-SITUATIONAL / NOT PRESENT),
    grounded in the diathesis-stress model.
    Applied per-item to produce ordinal PHQ-8 score 0-3.
    """
    prompt = (
        f"You are a clinical psychologist conducting a structured depression assessment.\n"
        f"You are evaluating ONE specific PHQ-8 symptom domain from a clinical interview.\n\n"

        f"PHQ-8 Item: {item_label}\n"
        f"Clinical definition: \"{item_hypothesis}\"\n"
        f"Participant answered: \"{answer_text}\"\n\n"

        f"CRITICAL CALIBRATION NOTE:\n"
        f"The participant is speaking conversationally, NOT filling out a clinical form. "
        f"They will almost never state explicit frequencies. "
        f"Infer severity from what the participant is claiming about their functioning, "
        f"not from how much they say, how emotionally they say it, or the presence or absence of specific words. "
        f"A short, flat statement and a long elaborated one can describe the same severity — "
        f"judge the claim, not the performance of the claim.\n\n"
 
        f"Analyze the participant's response step by step through five clinical stages. "
        f"Each stage must build on the previous stage's output — do not skip ahead.\n\n"
 

        # ── STAGE 1 ────────────────────────
        f"STAGE 1 — EMOTION ANALYSIS\n"
        f"Read the participant's entire response from start to finish before identifying anything. "
        f"The overall emotional meaning of a response is determined by its complete content, "
        f"not its opening clause. Identify:\n"
        f"  - Affective or somatic state described by the participant (emotional states such as mood, motivation, self-perception AND physical or functional states such as energy, sleep, appetite, movement)\n"
        f"  - Intensity: low / medium / high\n"
        f"  - Polarity: positive / negative / neutral / mixed\n"
        f"  - Source: internal (thoughts, feelings, self-perception) or "
        f"external (events, relationships, circumstances, others' observations)\n"
        f"  - Overall trajectory: does the response start positive and qualify negatively, "
        f"start negative and recover, or remain consistent throughout?\n"
        f"If the response contains any tension between two emotional states, "
        f"your emotion analysis must reflect BOTH states and their relative weight — "
        f"a response is only purely positive if every part of it is positive.\n\n"

       # ── STAGE 2 ───────────────────────────────
        f"STAGE 2 — CONTEXTUAL GROUNDING\n"
        f"Before deciding whether the symptom is present, identify the participant's life context. "
        f"This mirrors how a clinician opens a diagnostic interview: establish circumstances before probing symptoms. "
        f"Extract ONLY what the participant actually said or clearly implied — do not invent stressors.\n"
        f"  - External stressors: any recent events, losses, transitions, conflicts, or "
        f"ongoing strain (work, financial, health, caregiving, relationships) the participant mentions. "
        f"If none are mentioned or implied, state \"no external stressors identified.\"\n"
        f"  - Internal framing: does the participant describe the symptom as arising on its own — "
        f"without a clear trigger, persisting despite circumstances, or present before any stressor began? "
        f"If so, note this explicitly.\n"
        f"  - Stressor magnitude: for any external stressors identified, how much distress would a "
        f"typical person be expected to experience from them? (none / mild / moderate / severe)\n"
        f"This stage only gathers context. Do NOT decide here whether the symptom is internally or "
        f"externally caused — that judgement is made in Stage 3.\n\n"

        # ── STAGE 3 ───────────────
        f"STAGE 3 — ATTRIBUTION RESOLUTION\n"
        f"Using the emotion analysis from Stage 1 and the contextual grounding from Stage 2, "
        f"classify this specific PHQ-8 symptom into ONE of three categories:\n"
        f"  - PRESENT: the symptom is clearly endorsed and is NOT adequately explained by the stressors in Stage 2 — "
        f"either Stage 2 noted internal framing (the symptom arises on its own, persists despite circumstances, "
        f"or predates any stressor), or the symptom exceeds what the stressors' magnitude alone would explain.\n"
        f"  - PRESENT-SITUATIONAL: the symptom is endorsed but is proportionate to the external stressors identified in Stage 2, "
        f"judged against the stressor magnitude rated there — a typical person facing stressors of that magnitude "
        f"would be expected to experience similar distress, and Stage 2 noted no internal framing. "
        f"This captures situational distress (closer to an adjustment reaction) rather than an internal depressive process.\n"
        f"  - NOT PRESENT: the symptom is absent, denied, or contradicted by the participant's overall account.\n"
        f"This three-way scheme is grounded in the diathesis-stress model: distress that is fully accounted for "
        f"by external stress is clinically distinct from distress that exceeds or is independent of the stressor.\n\n"

        # ── STAGE 4 (REVISED — Reasoning Analysis, attribution-aware) ──────────
        f"STAGE 4 — REASONING ANALYSIS\n"
        f"Using the attribution from Stage 3, identify the underlying factors and how strongly each is expressed. "
        f"Preserve the intensity the participant conveyed — a total inability is not the same as an occasional difficulty, "
        f"and your Stage 4 output must carry that difference forward to Stage 5.\n"
        f"If PRESENT or PRESENT-SITUATIONAL: identify contributing factors across relevant dimensions —\n"
        f"  - Social: isolation, conflict, lack of support, relationship strain\n"
        f"  - Biological: sleep, energy, appetite, physical health\n"
        f"  - Psychological: guilt, worthlessness, hopelessness, self-perception\n"
        f"  - Functional: concentration, motivation, daily activities, work\n"
        f"For PRESENT-SITUATIONAL specifically: assess whether the symptom's intensity or pervasiveness "
        f"EXCEEDS what the stressors from Stage 2 — at the magnitude rated there — would alone explain. "
        f"Flag this explicitly.\n"
        f"If NOT PRESENT: identify protective factors — social support, coping mechanisms, "
        f"positive self-perception, healthy routines, resilience indicators.\n\n"

        # ── STAGE 5 (REVISED — Calibrated Severity Estimation) ─────────────────
        f"STAGE 5 — CALIBRATED SEVERITY ESTIMATION\n"
        f"Synthesise ALL information from Stages 1–4 to assign a PHQ-8 item score. "
        f"Your score should track the weight of what the prior stages produced. "
        f"A symptom classified as PRESENT or PRESENT-SITUATIONAL in Stage 3 can still be mild if Stage 4's factors are limited or Stage 1's emotional reading is contained — "
        f"and a symptom can be severe even without explicit frequency cues if the prior stages converge on a heavy, unresolved picture.\n\n"
        f"SCORING ACROSS ATTRIBUTION CATEGORIES:\n"
        f"  - PRESENT and PRESENT-SITUATIONAL cases: score on the full 0–3 scale based on the "
        f"symptom's actual intensity and functional impact. The attribution category records whether "
        f"an external stressor is involved; it does NOT raise or lower the score. Depression triggered "
        f"by external events is still depression and is scored on its severity, not its cause.\n"
        f"  - NOT PRESENT cases: score 0.\n\n"
        f"PHQ-8 anchors:\n"
        f"  0 = Not at all — symptom clearly absent; participant shows no indicators across the stages\n"
        f"  1 = Several days — mild or occasional; present but downplayed, fleeting, or with minimal functional impact\n"
        f"  2 = More than half the days — moderate; participant conveys this is a real and recurring problem that interferes with daily life, "
        f"even if they do not state an exact frequency\n"
        f"  3 = Nearly every day — severe; the symptom dominates the participant's experience "
        f"and causes significant impairment to their functioning\n\n"

        f"Respond ONLY with this exact JSON (no markdown):\n"
        f"{{\n"
        f"  \"stage1_emotion\": \"type, intensity, polarity, source, trajectory\",\n"
        f"  \"stage2_context\": \"external stressors, internal framing, stressor magnitude\",\n"
        f"  \"stage3_attribution\": \"PRESENT or PRESENT-SITUATIONAL or NOT PRESENT, with one-sentence justification\",\n"
        f"  \"stage4_reasoning\": \"contributing or protective factors by dimension; for PRESENT-SITUATIONAL flag whether symptom exceeds stressor\",\n"
        f"  \"stage5_score\": 0\n"
        f"}}"
    )
    try:
        raw    = llm.invoke(prompt).content.strip()
        raw    = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        score = int(result.get("stage5_score", 0))
        return max(0, min(3, score)), result
    except Exception:
        return 0, {}


# =============================================================================
#  STEP 8 — SEVERITY HELPER
# =============================================================================
def get_severity(score_value: int) -> str:
    if   score_value == 0:  return "No Depression"
    elif score_value <= 4:  return "Minimal Depression"
    elif score_value <= 9:  return "Mild Depression"
    elif score_value <= 14: return "Moderate Depression"
    elif score_value <= 19: return "Moderately Severe Depression"
    else:                   return "Severe Depression"


# =============================================================================
#  STEP 9 — AGENT STATE
# =============================================================================
class AgentState(TypedDict):
    # Conversation
    history:                    List[str]
    transcript:                 List[Dict]
    conversation_history_dicts: List[Dict]

    # Item tracking
    current_item_index:         int
    current_item_id:            str
    current_item_label:         str
    current_hypothesis:         str
    intro_turn_count:           int

    # Turn-level state
    last_question:              str
    last_answer:                str
    next_action:                str

    # GRM / IRT
    theta:                      float              # current MAP theta estimate
    item_responses:             Dict[str, int]     # phq_key → ordinal score 0-3
    asked_phq_keys:             List[str]
    grm_gain_log:               List[Dict]         # per-turn info-gain rankings
    pmi_last_gain:              float              # tracks previous turn best gain for stall detection

    # NLI evidence store (transcript record)
    items_evidence:             Dict[str, Any]

    # Output
    final_scores:               List[Dict]
    scoring_explanations:       List[Dict]
    agent_thoughts:             List[Dict]
    analytics_records:          List[Dict]
    symptom_summaries:          List[Dict]

    # Injected helpers
    _grm_params:                Any
    _pmi_matrix:                Any
    _item_entropy:              Any   # {phq_key: marginal H(X_j)} pre-computed from CSV

# =============================================================================
#  STEP 10 — LLM + PARSERS
# =============================================================================
llm        = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)
json_parser = JsonOutputParser()
str_parser  = StrOutputParser()

# =============================================================================
#  STEP 11 — AGENT PROMPTS
# =============================================================================

# ── Question Agent (initial item question)
question_template = ChatPromptTemplate.from_messages([
    ("system", """\
You are a warm, empathetic clinical interviewer conducting a PHQ-8 mental health check-in.
Your goal: ask ONE natural, conversational question about the symptom domain below.

Current estimated depression severity (theta): {theta:.2f}
  (theta < -1 = minimal, -1 to 0 = mild, 0 to 1 = moderate, > 1 = severe)

Current MIRT / evidence summary:
{evidence_summary}

Conversation so far:
<interview_history>
{history_text}
</interview_history>

Target domain (most informative given current theta):
  Domain : {domain}
  Meaning: {domain_meaning}
  Selection method: {loss_fn}

Output format — respond ONLY in this exact JSON (no markdown):
{{
  "question": "the single empathetic conversational question to ask",
  "reason": "one sentence on why this question fits the conversation now"
}}

Rules for the question:
1. Sound like a caring friend texting, not a clinician filling out a form.
2. EXACTLY ONE sentence. Always include the 2-week timeframe but weave it in naturally — 
   it should feel like part of the conversation, not a clinical checklist.
3. Natural ways to include the timeframe:
   "these past couple of weeks...", "lately — like the last few weeks...", 
   "over the past two weeks or so...", "recently, say the last two weeks..."
   Never say it robotically like "In the past 14 days have you..."
4. NO formal clinical openers like "To what extent...", "How would you rate..."
   Timeframe openers are allowed: "Over the last two weeks...", "In the past two weeks..."
5. Reference what the user just said if it fits naturally.
6. If theta > 0.5: probe with more specific, direct questions.
   If theta < -0.5: ask gently and broadly.
7. If theta > 1.0 ask specifically about frequency (days per week).
   If theta < -1.0 ask broadly and gently.

Tone examples:
  ❌ "Over the last couple of weeks, how often have you found yourself actually enjoying things?"
  ✅ "Have you still been enjoying stuff like that lately, or has it felt a bit flat?"
  ❌ "Have you been experiencing any changes in your sleep patterns recently?"
  ✅ "How's your sleep been? Falling asleep okay these days?"
"""),
    ("human", "Generate the next question."),
])


# =============================================================================
#  STEP 12 — GRAPH NODES
# =============================================================================

# ── NODE 1: Question Node
# LLM is called here. Selects the next most informative unasked item via the
# chosen loss function, then phrases the question naturally using question_template.
# No follow-up concept — every call to question_node is a fresh item question.
def question_node(state: AgentState):
    idx        = state["current_item_index"]
    theta      = state.get("theta", 0.0)
    current_id = state.get("current_item_id", "INTRO")

    # SCREENING — must be checked before idx==0 because both share idx=0
    if current_id == "SCREENING":
        state["current_item_label"] = SCREENING_LABEL
        state["current_hypothesis"] = SCREENING_HYPOTHESIS
        question = SCREENING_QUESTION

    # # INTRO — warm greeting, no scoring
    # elif idx == 0:
    #     state["current_item_id"]    = "INTRO"
    #     state["current_item_label"] = "Introduction"
    #     state["current_hypothesis"] = "Establish rapport."
    #     question = (
    #         "Hi, it's really nice to meet you. I'm here to have a relaxed conversation "
    #         "about how you've been feeling lately. There are no right or wrong answers — "
    #         "just share whatever feels comfortable. How are you doing today?"
    #     )

    # CLOSING
    elif idx == 9:
        state["current_item_id"]    = "CLOSING"
        state["current_item_label"] = "Closing"
        state["current_hypothesis"] = "End the interview."
        question = (
            "Thank you so much for taking the time to talk with me today. "
            "I really appreciate your openness. I hope you have a good rest of your day."
        )

    # PHQ-8 items — select next best item via loss function, ask via LLM
    else:
        grm_params   = state.get("_grm_params", {})
        pmi_matrix   = state.get("_pmi_matrix")
        asked_keys   = state.get("asked_phq_keys", [])
        responses    = state.get("item_responses", {})
        item_entropy = state.get("_item_entropy", {})

        ranked = select_next_item(
            theta=theta,
            asked_keys=asked_keys,
            responses=responses,
            grm_params=grm_params,
            pmi_matrix=pmi_matrix,
            item_entropy=item_entropy,
            loss_fn=LOSS_FUNCTION,
        )

        if not ranked:
            question = "Is there anything else about how you've been feeling you'd like to share?"
            current_phq_key = state.get("current_item_id", "")
        else:
            best_domain = ranked[0]
            phq_item    = PHQ_KEY_TO_ITEM.get(best_domain)
            if phq_item:
                state["current_item_id"]    = phq_item["item_id"]
                state["current_item_label"] = phq_item["label"]
                state["current_hypothesis"] = phq_item["text"]
                for i, h in enumerate(PHQ8_HYPOTHESES):
                    if h["item_id"] == phq_item["item_id"]:
                        state["current_item_index"] = i + 1
                        break
            current_phq_key = best_domain

            # Log gain scores
            gain_scores = {}
            for phq_key in ([best_domain] + ranked[1:3]):
                if LOSS_FUNCTION == "fisher":
                    gain_scores[phq_key] = fisher_information(theta, phq_key, grm_params)
                elif LOSS_FUNCTION == "pmi":
                    gain_scores[phq_key] = pmi_gain(phq_key, pmi_matrix, responses)
                elif LOSS_FUNCTION == "entropy":
                    gain_scores[phq_key] = entropy_gain(theta, phq_key, responses, grm_params, item_entropy)
            state["grm_gain_log"] = state.get("grm_gain_log", []) + [{
                "turn_index":      len(state["transcript"]) + 1,
                "theta":           round(theta, 4),
                "loss_function":   LOSS_FUNCTION,
                "selected_domain": best_domain,
                "gain_1":          round(gain_scores.get(best_domain, 0.0), 5),
                "candidate_2":     ranked[1] if len(ranked) > 1 else "",
                "gain_2":          round(gain_scores.get(ranked[1], 0.0), 5) if len(ranked) > 1 else 0.0,
                "candidate_3":     ranked[2] if len(ranked) > 2 else "",
                "gain_3":          round(gain_scores.get(ranked[2], 0.0), 5) if len(ranked) > 2 else 0.0,
                "remaining_count": len(ranked),
                "asked_count":     len(asked_keys),
            }]

            evidence_summary = "\n".join(
                f"  {k}: {round(v,4)} ({'strong' if v > 1.5 else 'weak'})"
                for k, v in sorted(responses.items(), key=lambda x: -x[1])
            )
            history_text = "\n".join(
                f"  {t['role'].upper()}: {t['content']}"
                for t in state.get("conversation_history_dicts", [])[-8:]
            ) or "  (start of interview)"

            raw_q = (question_template | llm | str_parser).invoke({
                "theta":            theta,
                "evidence_summary": evidence_summary,
                "history_text":     history_text,
                "domain":           current_phq_key,
                "domain_meaning":   PHQ8_CLINICAL_CONTEXT.get(current_phq_key, current_phq_key),
                "loss_fn":          LOSS_FUNCTION,
            }).strip().replace("```json","").replace("```","").strip()
            try:
                question = json.loads(raw_q).get("question", raw_q)
            except Exception:
                question = raw_q

    print(f"\n👩\u200d⚕️  Agent ({state.get('current_item_id','?')}) [θ={theta:.3f}]: {question}")

    conv_hist_dicts = state.get("conversation_history_dicts", []) + [{"role": "bot", "content": question}]
    turn = {
        "turn_index": len(state["transcript"]) + 1,
        "speaker":    AI_NAME,
        "text":       question,
        "role":       "question",
        "item_id":    state.get("current_item_id", "?"),
        "theta":      round(theta, 4),
    }
    return {
        "last_question":              question,
        "history":                    state["history"] + [f"Agent: {question}"],
        "transcript":                 state["transcript"] + [turn],
        "current_item_id":            state["current_item_id"],
        "current_item_label":         state["current_item_label"],
        "current_hypothesis":         state["current_hypothesis"],
        "current_item_index":         state.get("current_item_index", 0),
        "conversation_history_dicts": conv_hist_dicts,
        "grm_gain_log":               state.get("grm_gain_log", []),
    }


# ── NODE 2: Human Input Node
# Reads real participant answer from stdin.
# No LLM call for scoring — cosine similarity converts free text → ordinal 0-3.
# Then updates theta via GRM MAP estimation.
def human_input_node(state: AgentState):
    print(f"\n💬  {PARTICIPANT_NAME}: ", end="", flush=True)
    answer = input().strip()

    grm_params  = state.get("_grm_params", {})
    responses   = dict(state.get("item_responses", {}))
    asked_keys  = list(state.get("asked_phq_keys", []))
    theta       = state.get("theta", 0.0)
    current_id  = state.get("current_item_id", "INTRO")
    analytics   = list(state.get("analytics_records", []))

    if current_id == "SCREENING":
        # Score the broad screening answer to get initial theta estimate.
        # Uses a general depression proxy hypothesis — not a PHQ-8 item,
        # so it does NOT go into item_responses or asked_keys.
        # It only updates theta so the first PHQ-8 item is chosen adaptively.
        screening_score, _ = score_with_llm(
            answer_text=answer,
            item_label=SCREENING_LABEL,
            item_hypothesis=SCREENING_HYPOTHESIS,
            llm=llm,
        )
        print(f"   [LLM Score] Screening → score {screening_score}/3")
        # Use PHQ_8Depressed as proxy key — it has the highest discrimination
        # and the screening question maps most closely to general mood/depression
        screening_proxy = {"PHQ_8Depressed": screening_score}
        theta = estimate_theta_eap(screening_proxy, grm_params)
        print(f"   [GRM] θ initialised from screening → {theta:.3f}  (screening score: {screening_score}/3)")

    elif current_id not in ["INTRO", "CLOSING"]:
        item    = ITEM_ID_TO_ITEM.get(current_id, {})
        phq_key = item.get("phq_key", "")
        if phq_key:
            ordinal_score, cot_result = score_with_llm(
                answer_text=answer,
                item_label=item.get("label", ""),
                item_hypothesis=item.get("text", ""),
                llm=llm,
            )
            print(f"   [LLM Score] {phq_key} → score {ordinal_score}/3")
            print(f"   [CoT] Attribution={cot_result.get('stage3_attribution','?')} | "
                    f"Reasoning={cot_result.get('stage4_reasoning','?')[:80]}")
            responses[phq_key] = ordinal_score
            if phq_key not in asked_keys:
                asked_keys.append(phq_key)
            theta = estimate_theta_eap(responses, grm_params)
            print(f"   [GRM] θ={theta:.3f}  |  {phq_key} → {ordinal_score}/3")

            # ── Store full CoT reasoning for analytics
            analytics.append({
                "Item":               current_id,
                "PHQ_Key":            phq_key,
                "Answer":             answer,
                "Stage1_Emotion":     cot_result.get("stage1_emotion", ""),
                "Stage2_Context":     cot_result.get("stage2_context", ""),
                "Stage3_Attribution": cot_result.get("stage3_attribution", ""),
                "Stage4_Reasoning":   cot_result.get("stage4_reasoning", ""),
                "Stage5_Score":       ordinal_score,
            })

    conv_hist_dicts = state.get("conversation_history_dicts", []) + [{"role": "user", "content": answer}]
    turn = {
        "turn_index": len(state["transcript"]) + 1,
        "speaker":    PARTICIPANT_NAME,
        "text":       answer,
        "role":       "answer",
        "item_id":    current_id,
        "theta":      round(theta, 4),
    }
    return {
        "last_answer":                answer,
        "history":                    state["history"] + [f"Participant: {answer}"],
        "transcript":                 state["transcript"] + [turn],
        "conversation_history_dicts": conv_hist_dicts,
        "theta":                      theta,
        "item_responses":             responses,
        "asked_phq_keys":             asked_keys,
        "analytics_records":          analytics,
    }


# ── NODE 3: Navigation Node — pure GRM, no LLM, no follow-up concept
# Decision: NEXT_ITEM always.
# The only question is whether theta is well-estimated enough to move on,
# or whether information gain has been exhausted — both handled in transition_node.
# Navigation here just logs the current theta state and passes through.
THETA_SE_THRESHOLD = 0.6   # posterior SD — informational log only, not a gate

def _theta_posterior_sd(responses: Dict[str, int], grm_params: Dict) -> float:
    grid   = np.linspace(-4.0, 4.0, THETA_GRID_POINTS)
    log_w  = np.array([compute_log_posterior(t, responses, grm_params) for t in grid])
    log_w -= log_w.max()
    w      = np.exp(log_w); w /= w.sum()
    mean   = float(np.dot(w, grid))
    return float(np.sqrt(np.dot(w, (grid - mean) ** 2)))

def navigation_node(state: AgentState):
    current_id  = state.get("current_item_id", "INTRO")
    theta       = state.get("theta", 0.0)
    responses   = state.get("item_responses", {})
    grm_params  = state.get("_grm_params", {})
    last_answer = state.get("last_answer", "")
    asked_keys  = state.get("asked_phq_keys", [])
    items_data  = dict(state["items_evidence"])
    analytics   = list(state.get("analytics_records", []))
    thoughts    = list(state.get("agent_thoughts", []))

    posterior_sd = _theta_posterior_sd(responses, grm_params) if responses else 1.0

    icon = {"INTRO": "💬", "SCREENING": "🔍", "CLOSING": "🏁"}.get(current_id, "📋")
    print(f"   [GRM Nav] {icon} θ={theta:.3f} | SD={posterior_sd:.3f} | asked={len(asked_keys)}/8 → NEXT_ITEM")

    # Evidence tagging for transcript record
    if current_id not in ["INTRO", "SCREENING", "CLOSING"]:
        item_key = f"Item {state['current_item_index']}"
        if item_key in items_data:
            items_data[item_key]["neutral"].append({
                "text":         last_answer,
                "theta":        round(theta, 4),
                "posterior_sd": round(posterior_sd, 4),
                "parsed_score": responses.get(
                    ITEM_ID_TO_ITEM.get(current_id, {}).get("phq_key", ""), None
                ),
            })

    thoughts.append({
        "item":         current_id,
        "theta":        round(theta, 4),
        "posterior_sd": round(posterior_sd, 4),
        "asked_count":  len(asked_keys),
        "decision":     "NEXT_ITEM",
    })

    if current_id not in ["INTRO", "SCREENING", "CLOSING"]:
        pmi_matrix  = state.get("_pmi_matrix")
        current_phq = ITEM_ID_TO_ITEM.get(current_id, {}).get("phq_key", "")
        
        pmi_gain_val = (
            round(pmi_gain(current_phq, pmi_matrix, responses), 4)
            if LOSS_FUNCTION == "pmi" and pmi_matrix is not None and current_phq
            else None
        )

        pmi_marginal = (
            round(pmi_gain_val - state.get("pmi_last_gain", 0.0), 4) 
            if pmi_gain_val is not None else None
        )

        analytics.append({
            "Item":             current_id,
            "Theta":            round(theta, 4),
            "Posterior_SD":     round(posterior_sd, 4),
            "Parsed_Score":     responses.get(current_phq, -1),
            "Participant_Text": last_answer.replace('"', "'"),
            "Agent_Score":      -1,
            "PMI_Gain":         pmi_gain_val,
            "PMI_Marginal":     pmi_marginal,
            "Loss_Function":    LOSS_FUNCTION,
        })

    return {
        "next_action":       "NEXT_ITEM",
        "agent_thoughts":    thoughts,
        "items_evidence":    items_data,
        "analytics_records": analytics,
    }


# ── NODE 6: Transition Node — pure GRM, no LLM
# Decides which item to ask next (or close) based on information gain.
# Skips items whose gain < GAIN_THRESHOLDS[LOSS_FUNCTION] — already covered by theta.
def transition_node(state: AgentState):
    current_id = state.get("current_item_id", "INTRO")
    theta      = state.get("theta", 0.0)

    # # INTRO → SCREENING: after one warm greeting move to screening question
    # if current_id == "INTRO":
    #     return {
    #         "current_item_index": 0,          # still idx 0 — not a PHQ item yet
    #         "current_item_id":    "SCREENING",
    #         "current_item_label": SCREENING_LABEL,
    #         "current_hypothesis": SCREENING_HYPOTHESIS,
    #         "intro_turn_count":   1,
    #         "symptom_summaries":  state.get("symptom_summaries", []),
    #     }

    # SCREENING → first PHQ item, chosen adaptively from initialised theta
    if current_id == "SCREENING":
        # Theta is now initialised — select the most informative first item
        grm_params_s   = state.get("_grm_params", {})
        pmi_matrix_s   = state.get("_pmi_matrix")
        item_entropy_s = state.get("_item_entropy", {})
        theta_s        = state.get("theta", 0.0)
        responses_s    = state.get("item_responses", {})
        asked_s        = state.get("asked_phq_keys", [])

        ranked = select_next_item(
            theta=theta_s,
            asked_keys=asked_s,
            responses=responses_s,
            grm_params=grm_params_s,
            pmi_matrix=pmi_matrix_s,
            item_entropy=item_entropy_s,
            loss_fn=LOSS_FUNCTION,
        )
        first_item = PHQ_KEY_TO_ITEM.get(ranked[0]) if ranked else PHQ8_HYPOTHESES[0]
        print(f"   [GRM] First PHQ-8 item selected adaptively: {first_item['item_id']} "
              f"({first_item['label']}) at θ={theta_s:.3f}")
        return {
            "current_item_index": PHQ8_HYPOTHESES.index(first_item) + 1,
            "current_item_id":    first_item["item_id"],
            "current_item_label": first_item["label"],
            "current_hypothesis": first_item["text"],
            "symptom_summaries":  state.get("symptom_summaries", []),
        }

    # CLOSING
    if current_id == "CLOSING":
        return {
            "current_item_index": 10,
            "symptom_summaries":  state.get("symptom_summaries", []),
        }

    # Normal items — log symptom summary then pick next
    asked_keys   = list(state.get("asked_phq_keys", []))
    responses    = state.get("item_responses", {})
    grm_params   = state.get("_grm_params", {})
    pmi_matrix   = state.get("_pmi_matrix")
    item_entropy = state.get("_item_entropy", {})

    analytics    = list(state.get("analytics_records", []))
    current_logs = [r for r in analytics if r.get("Item") == current_id]
    symptom_entry = {
        "PID":          "PENDING",
        "Item":         current_id,
        "Theta_At_Item": round(theta, 4),
        "Parsed_Score": responses.get(
            ITEM_ID_TO_ITEM.get(current_id, {}).get("phq_key", ""), None
        ),
    }
    current_summaries = list(state.get("symptom_summaries", [])) + [symptom_entry]

    # Find eligible remaining items above gain threshold
    remaining = [h for h in PHQ8_HYPOTHESES if h["phq_key"] not in asked_keys]
    if not remaining:
        return {
            "current_item_index": 9,  "current_item_id": "CLOSING",
            "current_item_label": "Closing", "current_hypothesis": "End the conversation politely.",
            "symptom_summaries":  current_summaries,
        }
    
    best_gain = 0.0   # default — overwritten when LOSS_FUNCTION == "pmi"
    eligible = []
    if LOSS_FUNCTION == "pmi":
        # Compute best gain across all remaining items
        gains = {
            h["phq_key"]: pmi_gain(h["phq_key"], pmi_matrix, responses)
            for h in remaining
        }
        best_gain = max(gains.values()) if gains else 0.0
        last_gain = state.get("pmi_last_gain", 0.0)
        marginal  = best_gain - last_gain

        print(f"   [PMI] best_gain={best_gain:.4f} | last_gain={last_gain:.4f} | marginal={marginal:.4f}")

        if last_gain > 0 and marginal < PMI_STALL_THRESHOLD:
            print(f"   [PMI Stop] Gain stalled (Δ={marginal:.4f} < {PMI_STALL_THRESHOLD}) — closing.")
            return {
                "current_item_index": 9, "current_item_id": "CLOSING",
                "current_item_label": "Closing", "current_hypothesis": "End the conversation politely.",
                "symptom_summaries":  current_summaries,
                "pmi_last_gain":      best_gain,
            }
        else:
            eligible = remaining[:]
    else:
        # Fisher and Entropy: all remaining items eligible, stopping via SE(θ)
        eligible = remaining[:]

    if not eligible:
        return {
            "current_item_index": 9,  "current_item_id": "CLOSING",
            "current_item_label": "Closing", "current_hypothesis": "End the conversation politely.",
            "symptom_summaries":  current_summaries,
        }
    
    if LOSS_FUNCTION in ["fisher", "entropy"] and responses:
        posterior_sd = _theta_posterior_sd(responses, grm_params)
        if posterior_sd <= 0.3:
            print(f"   [GRM Stop] SE(θ)={posterior_sd:.3f} ≤ 0.20 — theta well-estimated, stopping early.")
            return {
                "current_item_index": 9,  "current_item_id": "CLOSING",
                "current_item_label": "Closing", "current_hypothesis": "End the conversation politely.",
                "symptom_summaries":  current_summaries,
            }

    next_item = eligible[0]
    return {
        "current_item_index": PHQ8_HYPOTHESES.index(next_item) + 1,
        "current_item_id":    next_item["item_id"],
        "current_item_label": next_item["label"],
        "current_hypothesis": next_item["text"],
        "symptom_summaries":  current_summaries,
        "pmi_last_gain": best_gain if LOSS_FUNCTION == "pmi" else state.get("pmi_last_gain", 0.0),
    }



# ── NODE 7: Batch Scoring Node — Pure ICC
# Scores ALL 8 items directly from the GRM Item Characteristic Curve
# LLM is called once per item to generate a clinical explanation of the score.
# at the final theta estimate — both items that were asked AND items that were
# skipped (low information gain). The skipped items are scored via prediction:
# since theta is a global latent trait shared across all items, the ICC gives
# P(X=k | theta) for every item regardless of whether it was directly asked.
# MAP score = argmax_k P(X=k | theta_final).
# Sufficiency = inverse of the distribution entropy — peaked = HIGH confidence.
def batch_scoring_node(state: AgentState):
    print("\n⏳ Interview complete. Scoring all 8 items via ICC (GRM) ...")

    theta      = state.get("theta", 0.0)
    grm_params = state.get("_grm_params", {})
    asked_keys = state.get("asked_phq_keys", [])

    print(f"   Final theta: {theta:.4f}")
    print(f"   Items directly asked : {len(asked_keys)}/8")
    print(f"   Items ICC-predicted  : {8 - len(asked_keys)}/8\n")

    final_scores         = []
    scoring_explanations = []
    updated_analytics    = list(state.get("analytics_records", []))


    # Build answer lookup: item_id → participant answer text
    answer_lookup = {}
    for turn in state.get("transcript", []):
        if turn.get("role") == "answer" and turn.get("item_id") not in ["INTRO", "SCREENING", "CLOSING"]:
            answer_lookup[turn["item_id"]] = turn["text"]

    # Full transcript answers for indirect evidence extraction
    all_answers = [
        {"item_id": t["item_id"], "text": t["text"]}
        for t in state.get("transcript", [])
        if t.get("role") == "answer" and t.get("item_id") not in ["INTRO", "SCREENING", "CLOSING"]
    ]

    for item_def in PHQ8_HYPOTHESES:
        item_id    = item_def["item_id"]
        item_label = item_def["label"]
        phq_key    = item_def["phq_key"]

        a      = grm_params[phq_key]["a"]
        b_list = grm_params[phq_key]["b"]

        # ── Full ICC probability distribution at final scoring theta
        # Returns [P(X=0), P(X=1), P(X=2), P(X=3)]
        probs = grm_category_probs(theta, a, b_list)

        # ── Score assignment:
        # If item was directly asked → use LLM score from item_responses
        # If item was skipped        → use ICC MAP score from final theta
        item_responses_state = state.get("item_responses", {})
        was_asked_early = phq_key in asked_keys
        if was_asked_early:
            score = item_responses_state.get(phq_key, int(np.argmax(probs)))
        else:
            score = int(round(sum(k * probs[k] for k in range(4))))

        # ── Confidence from distribution entropy
        # H = -Σ P(k) log P(k)
        # Low entropy = peaked distribution = HIGH confidence
        dist_entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        if   dist_entropy < 0.5:  sufficiency = "HIGH"
        elif dist_entropy < 1.0:  sufficiency = "MEDIUM"
        else:                     sufficiency = "LOW"

        was_asked = was_asked_early
        method    = "LLM (asked)" if was_asked else "ICC (predicted (item skipped))"
        icon      = "✅" if was_asked else "🔮"

        if was_asked:
            participant_answer = answer_lookup.get(item_id, "")
            evidence_context = f"The participant directly answered this question: \"{participant_answer}\""
        else:
            # Extract relevant evidence from other answers in transcript
            related_answers = "\n".join([
                f"- [{a['item_id']}]: {a['text']}"
                for a in all_answers
            ])
            participant_answer = "Not directly asked — inferred from related responses."
            evidence_context = (
                f"This item was not directly asked. "
                f"The following answers from other items informed the theta estimate "
                f"used to predict this score:\n{related_answers}"
            )

        explanation_prompt = (
            f"You are a clinical psychologist explaining a PHQ-8 scoring decision.\n"
            f"Item: {item_label} ({phq_key})\n"
            f"Domain: {PHQ8_CLINICAL_CONTEXT.get(phq_key, '')}\n\n"
            f"{evidence_context}\n\n"
            f"GRM theta: {round(theta, 3)}\n"
            f"ICC probabilities: P(0)={probs[0]:.3f}, P(1)={probs[1]:.3f}, "
            f"P(2)={probs[2]:.3f}, P(3)={probs[3]:.3f}\n"
            f"Final score: {score}/3\n\n"
            f"In 2-3 sentences, explain why this score was assigned. "
            f"{'Reference what the participant said directly.' if was_asked else 'Reference the related answers that informed the theta estimate and explain how they relate to this domain.'} "
            f"Be clinical but clear. Do not mention cosine similarity, GRM, or technical IRT details."
        )
        try:
            explanation = llm.invoke(explanation_prompt).content.strip()
        except Exception:
            explanation = f"Score {score} assigned based on ICC at θ={round(theta, 3)}."

        print(
            f"   {icon}  {item_id}: {score}/3 ({item_label})\n"
            f"        θ={theta:.3f} | a={a:.2f} | b={[round(b,2) for b in b_list]}\n"
            f"        P=[{', '.join(f'{p:.3f}' for p in probs)}]\n"
            f"        Entropy={dist_entropy:.3f} | Sufficiency={sufficiency} | {method}\n"
        )

        # ── Backfill analytics with final score
        for record in updated_analytics:
            if record["Item"] == item_id:
                record["Agent_Score"] = score   # LLM score if asked, ICC score if skipped

        final_scores.append({
            "Item ID":        item_id,
            "Item Label":     item_label,
            "Score":          score,
            "Sufficiency":    sufficiency,
            "Was_Asked":      was_asked,
            "Scoring_Method": method,
            "P0":             round(float(probs[0]), 4),
            "P1":             round(float(probs[1]), 4),
            "P2":             round(float(probs[2]), 4),
            "P3":             round(float(probs[3]), 4),
            "Dist_Entropy":   round(dist_entropy, 4),
        })

        scoring_explanations.append({
            "item_id":              item_id,
            "phq_key":              phq_key,
            "score":                score,
            "theta":                round(theta, 4),
            "discrimination_a":     round(a, 4),
            "thresholds_b":         [round(b, 4) for b in b_list],
            "probabilities": {
                "P(0)": round(float(probs[0]), 4),
                "P(1)": round(float(probs[1]), 4),
                "P(2)": round(float(probs[2]), 4),
                "P(3)": round(float(probs[3]), 4),
            },
            "dist_entropy":         round(dist_entropy, 4),
            "data_sufficiency":     sufficiency,
            "was_asked":            was_asked,
            "scoring_method":       method,
            "participant_evidence": participant_answer if was_asked else [
                {"item_id": a["item_id"], "text": a["text"]} for a in all_answers
            ],
            "evidence_type":        "direct" if was_asked else "indirect_from_transcript",
            "explanation":          explanation,
        })

    return {
        "final_scores":         final_scores,
        "scoring_explanations": scoring_explanations,
        "analytics_records":    updated_analytics,
    }

# =============================================================================
#  STEP 13 — GRAPH ASSEMBLY
# =============================================================================
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("question_node",      question_node)
    workflow.add_node("human_input_node",   human_input_node)
    workflow.add_node("navigation_node",    navigation_node)
    workflow.add_node("transition_node",    transition_node)
    workflow.add_node("batch_scoring_node", batch_scoring_node)

    # Linear flow — no follow-up branching
    # question → human → navigation → transition → question (next item) or scoring
    workflow.set_entry_point("question_node")
    def check_after_question(state):
        if state.get("current_item_id") == "CLOSING":
            return "transition_node"
        return "human_input_node"

    workflow.add_conditional_edges(
        "question_node", check_after_question,
        {"human_input_node": "human_input_node", "transition_node": "transition_node"}
    )
    workflow.add_edge("human_input_node", "navigation_node")
    workflow.add_edge("navigation_node",  "transition_node")

    def check_end(state):
        if state["current_item_index"] > 9:
            return "batch_scoring_node"
        return "question_node"

    workflow.add_conditional_edges(
        "transition_node", check_end,
        {"question_node": "question_node", "batch_scoring_node": "batch_scoring_node"},
    )

    workflow.add_edge("batch_scoring_node", END)
    return workflow.compile()


# =============================================================================
#  STEP 14 — MAIN
# =============================================================================
def main():
    global LOSS_FUNCTION   # declare before any use of LOSS_FUNCTION in this scope

    parser = argparse.ArgumentParser(description="MAGMA: Adaptive PHQ-8 Assessment")
    parser.add_argument("--id",   type=str, required=False,  help="Session identifier (e.g. P001, John, session_1) — used only to label output files")
    parser.add_argument("--loss", type=str, default=LOSS_FUNCTION,
                        choices=["fisher", "pmi", "entropy"],
                        help="Information-gain loss function (default: fisher)")
    args = parser.parse_args()

    # Override global loss function with CLI value
    LOSS_FUNCTION = args.loss
    print(f"\n📐 Loss function: {LOSS_FUNCTION.upper()}")

    # Build GRM parameters from real data
    print("Loading GRM parameters from dataset...")
    grm_params, pmi_matrix, item_entropy = build_grm_parameters(DATASET_PATH)

    print(f"\nGRM parameters (a = discrimination, b = thresholds):")
    for key, p in grm_params.items():
        print(f"  {key:25s}  a={p['a']:.3f}  b={[round(b,3) for b in p['b']]}")

    # NLI evidence store (MAGMA format)
    items_init = {
        f"Item {i+1}": {
            "label":         h["label"],
            "item_id":       h["item_id"],
            "supporting":    [],
            "contradicting": [],
            "neutral":       [],
        }
        for i, h in enumerate(PHQ8_HYPOTHESES)
    }

    state = {
        # Conversation
        "history":                    [],
        "transcript":                 [],
        "conversation_history_dicts": [],

        # Item tracking
        "current_item_index":         0,
        # "current_item_id":            "INTRO",
        # "current_item_label":         "Introduction",
        # "current_hypothesis":         "Establish rapport.",
        "current_item_id":            "SCREENING",
        "current_item_label":         SCREENING_LABEL,
        "current_hypothesis":         SCREENING_HYPOTHESIS,
        "intro_turn_count":           0,
        "intro_turn_count":           0,

        # Turn-level
        "last_question":              "",
        "last_answer":                "",

        # Clarification / alignment
                                                "next_action":                "",

        # IRT / GRM
        "theta":                      THETA_PRIOR_MEAN,
        "item_responses":             {},
        "asked_phq_keys":             [],
        "grm_gain_log":               [],
        "pmi_last_gain":              0.0,

        # Domain tracking
                                        
        # NLI evidence
        "items_evidence":             items_init,

        # Output
        "final_scores":               [],
        "scoring_explanations":       [],
        "agent_thoughts":             [],
        "analytics_records":          [],
        "symptom_summaries":          [],

        # Injected helpers
        "_grm_params":    grm_params,
        "_pmi_matrix":    pmi_matrix,
        "_item_entropy":  item_entropy,  # {phq_key: H(X_j)} pre-computed from CSV
    }

    print(f"\n🚀 MAGMA (ID: {args.id} | Loss: {LOSS_FUNCTION}) Starting...\n")
    app         = build_graph()
    final_state = app.invoke(state, {"recursion_limit": 500})

    # ==========================================================================
    #  SAVE FILES
    # ==========================================================================
    # Output folder structure:
    #   MAGMA/
    #     fisher/          ← or pmi/ or entropy/
    #       Evidence/
    #       Transcript/
    #       Agent_Thoughts/
    #       Scores/
    #       Scoring_Explanations/
    #       Analysis_Metrics/
    #       Symptoms/
    #       GRM_Gains/
    base_dir = os.path.join("MAGMA", LOSS_FUNCTION)
    dirs = {
        "ev": os.path.join(base_dir, "Evidence"),
        "tr": os.path.join(base_dir, "Transcript"),
        "th": os.path.join(base_dir, "Agent_Thoughts"),
        "sc": os.path.join(base_dir, "Scores"),
        "ex": os.path.join(base_dir, "Scoring_Explanations"),
        "an": os.path.join(base_dir, "Analysis_Metrics"),
        "sy": os.path.join(base_dir, "Symptoms"),
        "gn": os.path.join(base_dir, "GRM_Gains"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    pid = args.id

    # A. Evidence
    with open(os.path.join(dirs["ev"], f"Evidence_{pid}.json"), "w") as f:
        json.dump(final_state["items_evidence"], f, indent=2)

    # B. Transcript
    with open(os.path.join(dirs["tr"], f"Transcript_{pid}.jsonl"), "w") as f:
        for t in final_state["transcript"]:
            f.write(json.dumps(t) + "\n")

    # C. Agent Thoughts
    with open(os.path.join(dirs["th"], f"Thoughts_{pid}.jsonl"), "w") as f:
        for t in final_state["agent_thoughts"]:
            f.write(json.dumps(t) + "\n")

    # D. Scoring Explanations
    with open(os.path.join(dirs["ex"], f"Explanations_{pid}.json"), "w") as f:
        json.dump(final_state["scoring_explanations"], f, indent=2)

    # E. Scores CSV  (ICC Option A — all 8 items scored from theta via GRM)
    total_score  = sum(item["Score"] for item in final_state["final_scores"])
    severity_cat = get_severity(total_score)
    final_theta  = round(final_state.get("theta", 0.0), 4)
    asked_count  = len(final_state.get("asked_phq_keys", []))

    summary_rows = [
        {"Item ID": "TOTAL",       "Item Label": "PHQ-8 SUM",            "Score": total_score,
         "Sufficiency": "",        "Was_Asked": "", "Scoring_Method": "",
         "P0": "", "P1": "", "P2": "", "P3": "", "Dist_Entropy": ""},
        {"Item ID": "THETA",       "Item Label": "GRM Theta (final)",     "Score": final_theta,
         "Sufficiency": "",        "Was_Asked": "", "Scoring_Method": "GRM MAP",
         "P0": "", "P1": "", "P2": "", "P3": "", "Dist_Entropy": ""},
        {"Item ID": "SEVERITY",    "Item Label": "Severity Category",     "Score": severity_cat,
         "Sufficiency": "",        "Was_Asked": "", "Scoring_Method": "",
         "P0": "", "P1": "", "P2": "", "P3": "", "Dist_Entropy": ""},
        {"Item ID": "ITEMS_ASKED", "Item Label": "Items Directly Asked",  "Score": f"{asked_count}/8",
         "Sufficiency": "",        "Was_Asked": "", "Scoring_Method": "",
         "P0": "", "P1": "", "P2": "", "P3": "", "Dist_Entropy": ""},
        {"Item ID": "LOSS_FN",     "Item Label": "Loss Function Used",    "Score": LOSS_FUNCTION,
         "Sufficiency": "",        "Was_Asked": "", "Scoring_Method": "",
         "P0": "", "P1": "", "P2": "", "P3": "", "Dist_Entropy": ""},
    ]
    csv_data = final_state["final_scores"] + summary_rows
    csv_path = os.path.join(dirs["sc"], f"Scores_{pid}.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["Item ID", "Item Label", "Score", "Sufficiency",
                      "Was_Asked", "Scoring_Method",
                      "P0", "P1", "P2", "P3", "Dist_Entropy"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(csv_data)
    # F. Analytics
    analytics_path = os.path.join(dirs["an"], f"Analysis_{pid}.csv")
    records        = final_state.get("analytics_records", [])
    for r in records:
        r["ID"] = pid
    if records:
        # Use actual keys present in the records to avoid fieldname mismatch
        keys = ["ID", "Item", "PHQ_Key", "Answer", "Theta", "Posterior_SD", "Parsed_Score",
        "Agent_Score", "Participant_Text", "PMI_Gain", "PMI_Marginal", "Loss_Function",
        "Stage1_Emotion", "Stage2_Context", "Stage3_Attribution", "Stage4_Reasoning", "Stage5_Score"]
        with open(analytics_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(records)

    # G. Symptoms Summary
    symptoms_path = os.path.join(dirs["sy"], f"Symptoms_{pid}.csv")
    sym_records   = final_state.get("symptom_summaries", [])
    for r in sym_records:
        r["ID"] = pid
    if sym_records:
        with open(symptoms_path, "w", newline="") as f:
            keys = list(sym_records[0].keys())
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(sym_records)

    # H. GRM Gain Log
    gain_path    = os.path.join(dirs["gn"], f"GRM_Gains_{pid}.csv")
    gain_records = final_state.get("grm_gain_log", [])
    for r in gain_records:
        r["ID"] = pid
    if gain_records:
        keys = ["ID", "turn_index", "theta", "loss_function", "selected_domain",
                "gain_1", "candidate_2", "gain_2", "candidate_3", "gain_3",
                "remaining_count", "asked_count"]
        with open(gain_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(gain_records)

    print(f"\n✅ Done.")
    print(f"   Total PHQ-8 Score : {total_score} ({severity_cat})")
    print(f"   Final GRM theta   : {final_theta}")
    print(f"   Loss function used: {LOSS_FUNCTION.upper()}")
    print(f"   All files saved to: {os.path.abspath(base_dir)}")
    print(f"   Folder: MAGMA/{LOSS_FUNCTION}/")


if __name__ == "__main__":
    main()
