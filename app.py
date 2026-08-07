import os
import json
import uuid

import numpy as np
from flask import Flask, request, jsonify, Response

import MAGMA as M  

app = Flask(__name__)

# ---- Build GRM parameters ONCE at startup (girth MML fit is expensive) ----
print("Loading GRM parameters from dataset (one-time)...")
GRM_PARAMS, PMI_MATRIX, ITEM_ENTROPY = M.build_grm_parameters(M.DATASET_PATH)
print("GRM parameters cached. Server ready.")

SESSIONS = {} 

def _phrase_question(sess, phq_key):
    theta = sess["theta"]
    responses = sess["item_responses"]

    evidence_summary = "\n".join(
        f"  {k}: {round(v, 4)} ({'strong' if v > 1.5 else 'weak'})"
        for k, v in sorted(responses.items(), key=lambda x: -x[1])
    ) or "  (no items scored yet)"

    history_text = "\n".join(
        f"  {t['role'].upper()}: {t['text']}"
        for t in sess["transcript"][-8:]
    ) or "  (start of interview)"

    raw_q = (M.question_template | M.llm | M.str_parser).invoke({
        "theta":            theta,
        "evidence_summary": evidence_summary,
        "history_text":     history_text,
        "domain":           phq_key,
        "domain_meaning":   M.PHQ8_CLINICAL_CONTEXT.get(phq_key, phq_key),
        "loss_fn":          sess["loss"],
    }).strip().replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw_q).get("question", raw_q)
    except Exception:
        return raw_q


def _select_and_set_question(sess):
    ranked = M.select_next_item(
        theta=sess["theta"],
        asked_keys=sess["asked_phq_keys"],
        responses=sess["item_responses"],
        grm_params=GRM_PARAMS,
        pmi_matrix=PMI_MATRIX,
        item_entropy=ITEM_ENTROPY,
        loss_fn=sess["loss"],
    )
    if not ranked:
        return None

    phq_key = ranked[0]
    item = M.PHQ_KEY_TO_ITEM[phq_key]
    question = _phrase_question(sess, phq_key)

    sess["pending"] = {
        "type": "phq", "phq_key": phq_key, "item_id": item["item_id"],
        "label": item["label"], "hypothesis": item["text"],
    }
    sess["transcript"].append({"role": "agent", "text": question})
    return {
        "type": "question", "question": question,
        "item_id": item["item_id"], "label": item["label"],
        "theta": round(sess["theta"], 3), "asked": len(sess["asked_phq_keys"]),
    }


def _should_stop(sess):
    asked = sess["asked_phq_keys"]
    responses = sess["item_responses"]
    loss = sess["loss"]

    remaining = [h for h in M.PHQ8_HYPOTHESES if h["phq_key"] not in asked]
    if not remaining:
        return True

    if loss == "pmi":
        gains = {h["phq_key"]: M.pmi_gain(h["phq_key"], PMI_MATRIX, responses)
                 for h in remaining}
        best_gain = max(gains.values()) if gains else 0.0
        last_gain = sess.get("pmi_last_gain", 0.0)
        marginal = best_gain - last_gain
        sess["pmi_last_gain"] = best_gain
        return last_gain > 0 and marginal < M.PMI_STALL_THRESHOLD

    if responses:
        if M._theta_posterior_sd(responses, GRM_PARAMS) <= 0.3:
            return True
    return False


def _finalize(sess):
    theta = sess["theta"]
    asked = sess["asked_phq_keys"]
    responses = sess["item_responses"]

    items_out, total = [], 0
    for item_def in M.PHQ8_HYPOTHESES:
        phq_key = item_def["phq_key"]
        a = GRM_PARAMS[phq_key]["a"]
        b_list = GRM_PARAMS[phq_key]["b"]
        probs = M.grm_category_probs(theta, a, b_list)

        was_asked = phq_key in asked
        if was_asked:
            score = int(responses.get(phq_key, int(np.argmax(probs))))
        else:
            score = int(round(sum(k * probs[k] for k in range(4))))
        total += score

        dist_entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        sufficiency = "HIGH" if dist_entropy < 0.5 else ("MEDIUM" if dist_entropy < 1.0 else "LOW")

        items_out.append({
            "item_id": item_def["item_id"], "label": item_def["label"],
            "domain": M.PHQ8_CLINICAL_CONTEXT.get(phq_key, ""),
            "score": score, "was_asked": was_asked,
            "method": "LLM (asked)" if was_asked else "ICC (predicted)",
            "sufficiency": sufficiency,
            "probs": [round(float(p), 3) for p in probs],
        })

    sess["done"] = True
    sess["pending"] = None
    return {
        "type": "complete", "final_theta": round(theta, 3),
        "total_score": total, "severity": M.get_severity(total),
        "asked_count": len(asked), "skipped_count": 8 - len(asked),
        "loss": sess["loss"], "items": items_out,
    }

@app.route("/")
def index():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "index.html"), "r", encoding="utf-8") as f:
        return Response(f.read(), mimetype="text/html")


@app.route("/api/start", methods=["POST"])
def start():
    data = request.get_json(force=True) or {}
    loss = data.get("loss", "fisher")
    if loss not in ("fisher", "pmi", "entropy"):
        return jsonify({"error": "loss must be fisher | pmi | entropy"}), 400

    sid = uuid.uuid4().hex
    SESSIONS[sid] = {
        "loss": loss, "theta": M.THETA_PRIOR_MEAN,
        "item_responses": {}, "asked_phq_keys": [], "pmi_last_gain": 0.0,
        "transcript": [], "pending": {"type": "screening"}, "done": False,
    }
    sess = SESSIONS[sid]
    sess["transcript"].append({"role": "agent", "text": M.SCREENING_QUESTION})

    return jsonify({
        "session_id": sid, "type": "question",
        "question": M.SCREENING_QUESTION, "item_id": "SCREENING",
        "label": M.SCREENING_LABEL, "theta": round(sess["theta"], 3),
        "asked": 0, "loss": loss,
    })


@app.route("/api/answer", methods=["POST"])
def answer():
    data = request.get_json(force=True) or {}
    sid = data.get("session_id")
    text = (data.get("text") or "").strip()

    sess = SESSIONS.get(sid)
    if sess is None:
        return jsonify({"error": "Unknown session. Start a new simulation."}), 404
    if sess["done"]:
        return jsonify({"error": "This session is already complete."}), 400

    pending = sess["pending"]
    sess["transcript"].append({"role": "user", "text": text})

    # SCREENING -> init theta, then first adaptive PHQ item
    if pending["type"] == "screening":
        score, _ = M.score_with_llm(text, M.SCREENING_LABEL, M.SCREENING_HYPOTHESIS, M.llm)
        sess["theta"] = M.estimate_theta_eap({"PHQ_8Depressed": score}, GRM_PARAMS)
        turn = _select_and_set_question(sess)
        if turn is None:
            return jsonify({**_finalize(sess), "session_id": sid})
        return jsonify({**turn, "session_id": sid, "screening_score": score})

    # PHQ item -> score, update theta, decide stop vs next
    phq_key = pending["phq_key"]
    score, _ = M.score_with_llm(text, pending["label"], pending["hypothesis"], M.llm)
    sess["item_responses"][phq_key] = score
    if phq_key not in sess["asked_phq_keys"]:
        sess["asked_phq_keys"].append(phq_key)
    sess["theta"] = M.estimate_theta_eap(sess["item_responses"], GRM_PARAMS)

    if _should_stop(sess):
        return jsonify({**_finalize(sess), "session_id": sid,
                        "last_item": pending["item_id"], "last_score": score})

    turn = _select_and_set_question(sess)
    if turn is None:
        return jsonify({**_finalize(sess), "session_id": sid})
    return jsonify({**turn, "session_id": sid,
                    "last_item": pending["item_id"], "last_score": score})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
