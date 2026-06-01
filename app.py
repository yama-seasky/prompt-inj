"""
Flask web app for indirect prompt injection educational demo.
"""

import os
from flask import Flask, render_template, request, session, jsonify, redirect, url_for
from agent import run_agent

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "educational-demo-secret-key-2024")

EMAIL_LOG = os.path.join(os.path.dirname(__file__), "email_log.txt")
AGENT_LOG = os.path.join(os.path.dirname(__file__), "agent_log.txt")

# Hardcoded victim email for the demo
VICTIM_EMAIL = "victim@company.com"


@app.route("/")
def index():
    if "user_email" not in session:
        session["user_email"] = VICTIM_EMAIL
    if "conversation_history" not in session:
        session["conversation_history"] = []
    return render_template("index.html", user_email=session["user_email"])


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    user_message = data.get("message", "").strip()
    page_name = data.get("page_name", "product_safe")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Validate page_name to only allow known pages
    allowed_pages = {"product_safe", "product_injected"}
    if page_name not in allowed_pages:
        page_name = "product_safe"

    user_email = session.get("user_email", VICTIM_EMAIL)
    conversation_history = session.get("conversation_history", [])

    result = run_agent(
        user_message=user_message,
        page_name=page_name,
        user_email=user_email,
        conversation_history=conversation_history,
    )

    # Update conversation history (store last 10 turns to avoid session bloat)
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": result.get("response", "")})
    session["conversation_history"] = conversation_history[-20:]

    return jsonify({
        "response": result.get("response", ""),
        "tool_calls": result.get("tool_calls", []),
        "error": result.get("error"),
    })


@app.route("/logs")
def logs():
    email_log = ""
    agent_log = ""

    if os.path.exists(EMAIL_LOG):
        with open(EMAIL_LOG, "r", encoding="utf-8") as f:
            email_log = f.read()

    if os.path.exists(AGENT_LOG):
        with open(AGENT_LOG, "r", encoding="utf-8") as f:
            agent_log = f.read()

    return render_template("logs.html", email_log=email_log, agent_log=agent_log)


@app.route("/reset")
def reset():
    # Clear log files
    for log_file in [EMAIL_LOG, AGENT_LOG]:
        if os.path.exists(log_file):
            os.remove(log_file)
    # Clear session conversation history
    session["conversation_history"] = []
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
