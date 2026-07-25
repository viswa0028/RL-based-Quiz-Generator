# -*- coding: utf-8 -*-
"""
web_quiz.py

Flask version of the Adaptive Quiz System, now with:
  - Real user accounts (username/password, hashed with werkzeug).
  - Persistent per-user Q-table and concept-mastery, stored in SQLite
    (db.py) and reloaded on every login — the RL policy actually
    accumulates learning across sessions instead of resetting each run.
  - A /dashboard route showing quiz history and per-concept accuracy.
  - Escaped user input in results (no stored-XSS), debug off by default,
    secret key from environment.

Setup:
    pip install -r requirements.txt
    export GEMINI_API_KEY=your-key-here
    export FLASK_SECRET_KEY=some-random-string
    python web_quiz.py   # creates quiz_app.db on first run
"""

import os
import functools
import secrets
from collections import defaultdict
from markupsafe import escape

from flask import Flask, request, render_template_string, session, redirect, url_for
from dotenv import load_dotenv

import quiz_engine as qe
import db

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(16))
db.init_db()


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def login_required(view_func):
    @functools.wraps(view_func)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

BASE_STYLE = """
body { font-family: 'Poppins', sans-serif; background: linear-gradient(135deg, #74ebd5, #acb6e5);
       margin: 0; padding: 40px 20px; min-height: 100vh; box-sizing: border-box; }
.container { background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);
             max-width: 700px; margin: 0 auto; animation: fadeIn 0.5s ease-in; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
h1 { color: #2c3e50; text-align: center; margin-bottom: 10px; }
p { color: #7f8c8d; }
a { color: #3498db; }
input[type=text], input[type=password], textarea {
    width: 100%; padding: 12px; border: 2px solid #ecf0f1; border-radius: 10px;
    font-size: 16px; margin-bottom: 12px; box-sizing: border-box;
}
button { display: block; width: 100%; padding: 15px; background: #3498db; color: white; border: none;
         border-radius: 10px; font-size: 16px; cursor: pointer; transition: background 0.3s; }
button:hover { background: #2980b9; }
.error { color: #e74c3c; text-align: center; margin-bottom: 15px; }
.topnav { max-width: 700px; margin: 0 auto 15px auto; display: flex; justify-content: space-between;
          color: white; font-size: 14px; }
.topnav a { color: white; text-decoration: underline; margin-left: 12px; }
table { width: 100%; border-collapse: collapse; margin-top: 10px; }
th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1; font-size: 14px; }
.stat-box { display: inline-block; background: #f9f9f9; border-radius: 10px; padding: 15px 20px; margin: 5px; }
"""

LOGIN_TEMPLATE = """
<!DOCTYPE html><html><head><title>Log in</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
<style>{{ base_style }}</style></head><body>
<div class="container">
<h1>Adaptive Quiz System</h1>
<p style="text-align:center;">Log in to track your progress</p>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
<form method="POST">
<input type="text" name="username" placeholder="Username" required>
<input type="password" name="password" placeholder="Password" required>
<button type="submit">Log In</button>
</form>
<p style="text-align:center;">No account? <a href="{{ url_for('register') }}">Register</a></p>
</div>
</body></html>
"""

REGISTER_TEMPLATE = """
<!DOCTYPE html><html><head><title>Register</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
<style>{{ base_style }}</style></head><body>
<div class="container">
<h1>Create an account</h1>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
<form method="POST">
<input type="text" name="username" placeholder="Choose a username" required>
<input type="password" name="password" placeholder="Choose a password" required>
<button type="submit">Register</button>
</form>
<p style="text-align:center;">Already have an account? <a href="{{ url_for('login') }}">Log in</a></p>
</div>
</body></html>
"""

HOME_TEMPLATE = """
<!DOCTYPE html><html><head><title>Adaptive Quiz System</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
<style>{{ base_style }}</style></head><body>
<div class="topnav">
  <span>Logged in as {{ username }}</span>
  <span><a href="{{ url_for('dashboard') }}">Dashboard</a><a href="{{ url_for('logout') }}">Log out</a></span>
</div>
<div class="container">
<h1>Adaptive Quiz System</h1>
<p style="text-align:center;">Current difficulty: <strong>{{ difficulty }}</strong></p>
<p>Enter your text below to generate a custom quiz!</p>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
<form method="POST" action="{{ url_for('quiz') }}">
<textarea name="text" rows="6" placeholder="Type or paste your text here..." required></textarea>
<button type="submit">Generate Quiz</button>
</form>
</div>
</body></html>
"""

QUIZ_TEMPLATE = """
<!DOCTYPE html><html><head><title>Adaptive Quiz</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
<style>{{ base_style }}
.question { background: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
.question p { color: #34495e; margin: 0 0 15px 0; font-weight: 600; }
.options label { display: block; padding: 10px; background: #ecf0f1; margin: 5px 0; border-radius: 8px; cursor: pointer; }
.options input[type="radio"] { margin-right: 10px; }
</style></head><body>
<div class="container">
<h1>Adaptive Quiz</h1>
<p style="text-align:center;">Difficulty: {{ difficulty }}</p>
<form method="POST" action="{{ url_for('submit') }}">
{% for i in range(questions|length) %}
<div class="question">
<p>Q{{ i + 1 }}. [{{ concepts[i] }}] {{ questions[i] }}</p>
{% if is_mcq_flags[i] %}
<div class="options">
{% set options, correct = answers[i] %}
<label><input type="radio" name="answer_{{ i }}" value="A" required> A. {{ options['A'] }}</label>
<label><input type="radio" name="answer_{{ i }}" value="B"> B. {{ options['B'] }}</label>
<label><input type="radio" name="answer_{{ i }}" value="C"> C. {{ options['C'] }}</label>
<label><input type="radio" name="answer_{{ i }}" value="D"> D. {{ options['D'] }}</label>
</div>
{% else %}
<textarea name="answer_{{ i }}" rows="3" placeholder="Type your answer here..." required></textarea>
{% endif %}
</div>
{% endfor %}
<button type="submit">Submit Answers</button>
</form>
</div>
</body></html>
"""

RESULT_TEMPLATE = """
<!DOCTYPE html><html><head><title>Quiz Result</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
<style>{{ base_style }}
.score { font-size: 36px; color: #3498db; font-weight: 600; text-align:center; margin-bottom: 20px; }
.feedback-item { margin-bottom: 15px; padding: 10px; border-left: 4px solid #e74c3c; background: #f9f9f9; }
</style></head><body>
<div class="container">
<h1>Quiz Result</h1>
<div class="score">{{ score }} / 10</div>
<p>{{ summary_line }}</p>
{% if feedback_details %}
<h3 style="color:#e74c3c;">Areas for Improvement</h3>
{% for d in feedback_details %}
<div class="feedback-item">
<p><strong>Concept:</strong> {{ d.concept }}</p>
<p><strong>Question:</strong> {{ d.question }}</p>
<p><strong>Your Answer:</strong> {{ d.user_answer }}</p>
<p><strong>Expected Answer:</strong> {{ d.expected }}</p>
<p><strong>Feedback:</strong> {{ d.feedback }}</p>
</div>
{% endfor %}
{% else %}
<p>No significant weaknesses identified this round.</p>
{% endif %}
<p><strong>Next difficulty:</strong> {{ next_difficulty }}</p>
<form method="GET" action="{{ url_for('home') }}"><button type="submit">Try Another Quiz</button></form>
</div>
</body></html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html><html><head><title>Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
<style>{{ base_style }}</style></head><body>
<div class="topnav">
  <span>Logged in as {{ username }}</span>
  <span><a href="{{ url_for('home') }}">New quiz</a><a href="{{ url_for('logout') }}">Log out</a></span>
</div>
<div class="container">
<h1>Your Progress</h1>
{% if summary is none %}
<p>No quizzes yet — <a href="{{ url_for('home') }}">take your first one</a>.</p>
{% else %}
<div>
  <span class="stat-box"><strong>{{ summary.total_attempts }}</strong><br>quizzes taken</span>
  <span class="stat-box"><strong>{{ summary.avg_score }}/10</strong><br>average score</span>
  {% if summary.best_concept %}<span class="stat-box"><strong>{{ summary.best_concept }}</strong><br>strongest concept</span>{% endif %}
  {% if summary.worst_concept %}<span class="stat-box"><strong>{{ summary.worst_concept }}</strong><br>needs work</span>{% endif %}
</div>
<h3>Recent attempts</h3>
<table>
<tr><th>Date</th><th>Difficulty</th><th>Score</th></tr>
{% for a in summary.recent_attempts %}
<tr><td>{{ a.created_at[:16].replace('T', ' ') }}</td><td>{{ a.difficulty }}</td><td>{{ a.score }}/10</td></tr>
{% endfor %}
</table>
{% endif %}
</div>
</body></html>
"""


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template_string(REGISTER_TEMPLATE, base_style=BASE_STYLE, error=None)

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    if not username or not password:
        return render_template_string(
            REGISTER_TEMPLATE, base_style=BASE_STYLE, error="Username and password are required."
        )
    try:
        user_id = db.create_user(username, password)
    except ValueError as e:
        return render_template_string(REGISTER_TEMPLATE, base_style=BASE_STYLE, error=str(e))

    session["user_id"] = user_id
    return redirect(url_for("home"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template_string(LOGIN_TEMPLATE, base_style=BASE_STYLE, error=None)

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    user_id = db.verify_user(username, password)
    if user_id is None:
        return render_template_string(
            LOGIN_TEMPLATE, base_style=BASE_STYLE, error="Incorrect username or password."
        )
    session["user_id"] = user_id
    return redirect(url_for("home"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Quiz routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
@login_required
def home():
    state = db.get_user_state(session["user_id"])
    return render_template_string(
        HOME_TEMPLATE,
        base_style=BASE_STYLE,
        username=db.get_username(session["user_id"]),
        difficulty=qe.DIFFICULTY_LEVELS[state["current_difficulty"]],
        error=None,
    )


@app.route("/quiz", methods=["POST"])
@login_required
def quiz():
    text = request.form.get("text", "")
    if not text.strip():
        state = db.get_user_state(session["user_id"])
        return render_template_string(
            HOME_TEMPLATE,
            base_style=BASE_STYLE,
            username=db.get_username(session["user_id"]),
            difficulty=qe.DIFFICULTY_LEVELS[state["current_difficulty"]],
            error="Text cannot be empty.",
        ), 400

    state = db.get_user_state(session["user_id"])
    model = qe.setup_model()
    questions, answers, concepts, is_mcq_flags = qe.generate_questions(
        model, state["current_difficulty"], text
    )

    # Stash the generated quiz in the session (small payload, fine to keep
    # client-side); the durable RL/mastery state lives in SQLite, not here.
    session["quiz_questions"] = questions
    session["quiz_answers"] = answers
    session["quiz_concepts"] = concepts
    session["quiz_is_mcq"] = is_mcq_flags

    return render_template_string(
        QUIZ_TEMPLATE,
        base_style=BASE_STYLE,
        difficulty=qe.DIFFICULTY_LEVELS[state["current_difficulty"]],
        questions=questions,
        answers=answers,
        concepts=concepts,
        is_mcq_flags=is_mcq_flags,
    )


@app.route("/submit", methods=["POST"])
@login_required
def submit():
    user_id = session["user_id"]
    questions = session.get("quiz_questions", [])
    answers = session.get("quiz_answers", [])
    concepts = session.get("quiz_concepts", [])
    is_mcq_flags = session.get("quiz_is_mcq", [])

    if not questions:
        return redirect(url_for("home"))

    state = db.get_user_state(user_id)
    q_table = state["q_table"]
    concept_mastery = state["concept_mastery"]
    current_difficulty = state["current_difficulty"]
    previous_score = state["previous_score"]

    model = qe.setup_model()
    user_answers = [request.form.get(f"answer_{i}", "") for i in range(len(questions))]

    total_score = 0
    concept_results = defaultdict(list)
    feedback_details = []

    for question, user_answer, answer_data, concept, is_mcq in zip(
        questions, user_answers, answers, concepts, is_mcq_flags
    ):
        if is_mcq:
            options, correct_answer = answer_data
            score = qe.evaluate_mcq_answer(user_answer, correct_answer)
            feedback = "Correct" if score == 1 else f"Incorrect (correct answer: {correct_answer})"
            expected = f"Correct: {correct_answer}"
        else:
            sample_answer = answer_data
            score, feedback = qe.evaluate_text_answer(
                model, question, user_answer, sample_answer, concept
            )
            expected = sample_answer

        concept_results[concept].append(score)
        total_score += score

        threshold = 1 if is_mcq else 0.5
        if score < threshold:
            feedback_details.append(
                {
                    "concept": escape(concept),
                    "question": escape(question),
                    "user_answer": escape(user_answer),
                    "expected": escape(expected),
                    "feedback": escape(feedback),
                }
            )

    for concept, results in concept_results.items():
        concept_mastery[concept].extend(results)

    performance_bucket = qe.get_performance_bucket(previous_score)
    mastery_level = qe.get_mastery_level(list(concept_results.keys()), concept_mastery)
    current_state_index = qe.get_state_index(current_difficulty, performance_bucket, mastery_level)

    action = qe.choose_action(q_table, current_state_index)
    next_difficulty = qe.adjust_difficulty(current_difficulty, action)

    reward = qe.calculate_reward(total_score, previous_score, current_difficulty)
    next_state_index = qe.get_state_index(
        next_difficulty,
        qe.get_performance_bucket(total_score),
        qe.get_mastery_level(list(concept_results.keys()), concept_mastery),
    )
    qe.update_q_table(q_table, current_state_index, action, reward, next_state_index)

    # Persist everything: this is what survives across logins/restarts.
    db.save_user_state(user_id, q_table, concept_mastery, next_difficulty, total_score)
    db.record_attempt(
        user_id,
        qe.DIFFICULTY_LEVELS[current_difficulty],
        total_score,
        {c: r for c, r in concept_results.items()},
    )

    if total_score >= 8:
        summary_line = "Excellent work! You scored high and demonstrated strong understanding."
    elif total_score >= 5:
        summary_line = "Good effort! You have a solid grasp of many concepts, with some areas to improve."
    else:
        summary_line = "Keep practicing! There are several areas where you can improve your understanding."

    return render_template_string(
        RESULT_TEMPLATE,
        base_style=BASE_STYLE,
        score=total_score,
        summary_line=summary_line,
        feedback_details=feedback_details,
        next_difficulty=qe.DIFFICULTY_LEVELS[next_difficulty],
    )


@app.route("/dashboard")
@login_required
def dashboard():
    summary = db.get_dashboard_summary(session["user_id"])
    return render_template_string(
        DASHBOARD_TEMPLATE,
        base_style=BASE_STYLE,
        username=db.get_username(session["user_id"]),
        summary=summary,
    )


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode, port=int(os.environ.get("PORT", 5000)))