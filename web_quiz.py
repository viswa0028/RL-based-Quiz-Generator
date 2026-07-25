# -*- coding: utf-8 -*-
"""
web_quiz.py

Flask version of the Adaptive Quiz System.

Fixes vs. the original "website for quiz generator.py":
  - No hardcoded API key (reads GEMINI_API_KEY from environment/.env).
  - Per-user state via Flask session instead of module-level globals
    (the original used one shared `quiz_state`/`current_difficulty` for
    every visitor, so two people using the app at once would corrupt
    each other's quiz).
  - User-supplied text (answers) is escaped before being rendered, since
    the original built the result page HTML by hand with `|safe`, which
    is a stored XSS vector.
  - debug mode is off unless FLASK_DEBUG=1 is set explicitly.
  - No threading/notebook bootstrap hacks; run with `python web_quiz.py`
    or a proper WSGI server.

Setup:
    pip install -r requirements.txt
    export GEMINI_API_KEY=your-key-here   # or put it in a .env file
    export FLASK_SECRET_KEY=some-random-string

Run:
    python web_quiz.py
"""

import os
import secrets
from collections import defaultdict
from markupsafe import escape

from flask import Flask, request, render_template_string, session
from dotenv import load_dotenv

import quiz_engine as qe

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(16))

# concept_mastery tracks accuracy per concept across a session's quizzes.
# Kept server-side keyed by a per-session id rather than one shared global.
_concept_mastery_by_session = defaultdict(lambda: defaultdict(list))
_q_table_by_session = {}


def _session_id():
    if "sid" not in session:
        session["sid"] = secrets.token_hex(8)
    return session["sid"]


def _concept_mastery():
    return _concept_mastery_by_session[_session_id()]


def _q_table():
    sid = _session_id()
    if sid not in _q_table_by_session:
        _q_table_by_session[sid] = qe.new_q_table()
    return _q_table_by_session[sid]


HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Adaptive Quiz System</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
<style>
body { font-family: 'Poppins', sans-serif; background: linear-gradient(135deg, #74ebd5, #acb6e5);
       margin: 0; padding: 0; min-height: 100vh; display: flex; justify-content: center; align-items: center; }
.container { background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);
             width: 90%; max-width: 600px; animation: fadeIn 0.5s ease-in; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
h1 { color: #2c3e50; text-align: center; margin-bottom: 20px; font-weight: 600; }
p { color: #7f8c8d; text-align: center; margin-bottom: 30px; }
textarea { width: 100%; height: 150px; padding: 15px; border: 2px solid #ecf0f1; border-radius: 10px;
           resize: none; font-size: 16px; transition: border-color 0.3s; box-sizing: border-box; }
textarea:focus { border-color: #3498db; outline: none; }
button { display: block; width: 100%; padding: 15px; background: #3498db; color: white; border: none;
         border-radius: 10px; font-size: 16px; cursor: pointer; transition: background 0.3s, transform 0.2s; }
button:hover { background: #2980b9; transform: translateY(-2px); }
.error { color: #e74c3c; text-align: center; margin-bottom: 15px; }
</style>
</head>
<body>
<div class="container">
<h1>Adaptive Quiz System</h1>
<p>Enter your text below to generate a custom quiz!</p>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
<form method="POST" action="/quiz">
<textarea name="text" placeholder="Type or paste your text here..." required></textarea>
<button type="submit">Generate Quiz</button>
</form>
</div>
</body>
</html>
"""

QUIZ_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Adaptive Quiz</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
<style>
body { font-family: 'Poppins', sans-serif; background: linear-gradient(135deg, #74ebd5, #acb6e5);
       margin: 0; padding: 40px 20px; min-height: 100vh; }
.container { background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);
             max-width: 800px; margin: 0 auto; animation: fadeIn 0.5s ease-in; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
h1 { color: #2c3e50; text-align: center; margin-bottom: 10px; }
.difficulty { text-align: center; color: #7f8c8d; margin-bottom: 30px; font-size: 18px; }
.question { background: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px; transition: transform 0.2s; }
.question:hover { transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.05); }
.question p { color: #34495e; margin: 0 0 15px 0; font-weight: 600; }
.options label { display: block; padding: 10px; background: #ecf0f1; margin: 5px 0; border-radius: 8px;
                  cursor: pointer; transition: background 0.3s; }
.options input[type="radio"] { margin-right: 10px; }
.options label:hover { background: #dfe6e9; }
textarea { width: 100%; padding: 15px; border: 2px solid #ecf0f1; border-radius: 10px; font-size: 16px;
           resize: vertical; transition: border-color 0.3s; box-sizing: border-box; }
textarea:focus { border-color: #3498db; outline: none; }
button { display: block; width: 100%; padding: 15px; background: #3498db; color: white; border: none;
         border-radius: 10px; font-size: 16px; cursor: pointer; transition: background 0.3s, transform 0.2s; }
button:hover { background: #2980b9; transform: translateY(-2px); }
</style>
</head>
<body>
<div class="container">
<h1>Adaptive Quiz</h1>
<p class="difficulty">Difficulty: {{ difficulty }}</p>
<form method="POST" action="/submit">
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
<textarea name="answer_{{ i }}" placeholder="Type your answer here..." required></textarea>
{% endif %}
</div>
{% endfor %}
<button type="submit">Submit Answers</button>
</form>
</div>
</body>
</html>
"""

RESULT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Quiz Result</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
<style>
body { font-family: 'Poppins', sans-serif; background: linear-gradient(135deg, #74ebd5, #acb6e5);
       margin: 0; padding: 40px 20px; min-height: 100vh; display: flex; justify-content: center; align-items: center; }
.container { background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);
             max-width: 800px; width: 90%; text-align: center; animation: fadeIn 0.5s ease-in; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
h1 { color: #2c3e50; margin-bottom: 20px; }
.score { font-size: 36px; color: #3498db; font-weight: 600; margin-bottom: 20px; }
.summary { color: #34495e; font-size: 16px; text-align: left; margin-bottom: 20px; }
.feedback-item { margin-bottom: 15px; padding: 10px; border-left: 4px solid #e74c3c; background: #f9f9f9; text-align: left; }
.feedback-item p { margin: 5px 0; }
.next-difficulty { color: #34495e; font-size: 18px; margin-bottom: 30px; }
button { padding: 15px 30px; background: #3498db; color: white; border: none; border-radius: 10px;
         font-size: 16px; cursor: pointer; transition: background 0.3s, transform 0.2s; }
button:hover { background: #2980b9; transform: translateY(-2px); }
</style>
</head>
<body>
<div class="container">
<h1>Quiz Result</h1>
<div class="score">{{ score }} / 10</div>
<div class="summary"><p>{{ summary_line }}</p></div>
{% if feedback_details %}
<h3 style="text-align:left; color:#e74c3c;">Areas for Improvement</h3>
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
<p>Congratulations! No significant weaknesses identified.</p>
{% endif %}
<p class="next-difficulty">Next Difficulty: {{ next_difficulty }}</p>
<form method="GET" action="/">
<button type="submit">Try Another Quiz</button>
</form>
</div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HOME_TEMPLATE, error=None)


@app.route("/quiz", methods=["POST"])
def quiz():
    text = request.form.get("text", "")
    if not text.strip():
        return render_template_string(HOME_TEMPLATE, error="Text cannot be empty."), 400

    current_difficulty = session.get("current_difficulty", 1)
    model = qe.setup_model()
    questions, answers, concepts, is_mcq_flags = qe.generate_questions(
        model, current_difficulty, text
    )

    session["questions"] = questions
    session["answers"] = answers
    session["concepts"] = concepts
    session["is_mcq_flags"] = is_mcq_flags
    session["current_difficulty"] = current_difficulty
    session.setdefault("previous_score", 5)

    return render_template_string(
        QUIZ_TEMPLATE,
        difficulty=qe.DIFFICULTY_LEVELS[current_difficulty],
        questions=questions,
        answers=answers,
        concepts=concepts,
        is_mcq_flags=is_mcq_flags,
    )


@app.route("/submit", methods=["POST"])
def submit():
    questions = session.get("questions", [])
    answers = session.get("answers", [])
    concepts = session.get("concepts", [])
    is_mcq_flags = session.get("is_mcq_flags", [])
    current_difficulty = session.get("current_difficulty", 1)
    previous_score = session.get("previous_score", 5)

    if not questions:
        return render_template_string(HOME_TEMPLATE, error="Please start a new quiz first."), 400

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
                    # escape() prevents a submitted answer like "<script>..."
                    # from being rendered as live HTML (stored XSS).
                    "concept": escape(concept),
                    "question": escape(question),
                    "user_answer": escape(user_answer),
                    "expected": escape(expected),
                    "feedback": escape(feedback),
                }
            )

    mastery_store = _concept_mastery()
    for concept, results in concept_results.items():
        mastery_store[concept].extend(results)

    q_table = _q_table()
    performance_bucket = qe.get_performance_bucket(previous_score)
    mastery_level = qe.get_mastery_level(list(concept_results.keys()), mastery_store)
    current_state_index = qe.get_state_index(current_difficulty, performance_bucket, mastery_level)

    action = qe.choose_action(q_table, current_state_index)
    next_difficulty = qe.adjust_difficulty(current_difficulty, action)

    reward = qe.calculate_reward(total_score, previous_score, current_difficulty)
    next_state_index = qe.get_state_index(
        next_difficulty,
        qe.get_performance_bucket(total_score),
        qe.get_mastery_level(list(concept_results.keys()), mastery_store),
    )
    qe.update_q_table(q_table, current_state_index, action, reward, next_state_index)

    if total_score >= 8:
        summary_line = "Excellent work! You scored high and demonstrated strong understanding."
    elif total_score >= 5:
        summary_line = "Good effort! You have a solid grasp of many concepts, with some areas to improve."
    else:
        summary_line = "Keep practicing! There are several areas where you can improve your understanding."

    session["previous_score"] = total_score
    session["current_difficulty"] = next_difficulty

    return render_template_string(
        RESULT_TEMPLATE,
        score=total_score,
        summary_line=summary_line,
        feedback_details=feedback_details,
        next_difficulty=qe.DIFFICULTY_LEVELS[next_difficulty],
    )


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode, port=int(os.environ.get("PORT", 5000)))
