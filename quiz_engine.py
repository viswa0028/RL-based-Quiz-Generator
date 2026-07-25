# -*- coding: utf-8 -*-
"""
quiz_engine.py

Shared core logic for the Adaptive Quiz System: RL-based difficulty
selection, question generation via Gemini, answer evaluation, and
translation support.

Both `cli_quiz.py` (terminal version) and `web_quiz.py` (Flask version)
import from this module instead of duplicating the logic. This is the
only place the Q-learning, prompting, and parsing code should live.
"""

import os
import re
import json
import random
import logging
from collections import defaultdict

import numpy as np
import google.generativeai as genai

try:
    from googletrans import Translator
    _translator = Translator()
except Exception:  # pragma: no cover - translation is optional
    _translator = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quiz_engine")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
NUM_DIFFICULTIES = len(DIFFICULTY_LEVELS)
NUM_PERFORMANCE_BUCKETS = 3
NUM_MASTERY_LEVELS = 3
STATE_SPACE_SIZE = NUM_DIFFICULTIES * NUM_PERFORMANCE_BUCKETS * NUM_MASTERY_LEVELS
NUM_ACTIONS = 3  # 0 = decrease, 1 = keep, 2 = increase

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
DEFAULT_EPSILON = 0.3  # exploration rate for the RL policy

DIFFICULTY_DESCRIPTIONS = {
    "easy": "simple facts",
    "medium": "processes",
    "hard": "complex applications",
}


def configure_api():
    """
    Reads the API key from the GEMINI_API_KEY environment variable.
    Never hardcode API keys in source files that get committed to git.

    Raises a clear error instead of silently configuring genai with a
    placeholder string, which was the previous behavior.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Create a .env file (see .env.example) or export it in your shell: "
            "export GEMINI_API_KEY=your-key-here"
        )
    genai.configure(api_key=api_key)


def setup_model():
    configure_api()
    return genai.GenerativeModel("gemini-1.5-flash")


# ---------------------------------------------------------------------------
# Q-learning state/action helpers
# ---------------------------------------------------------------------------

def new_q_table():
    """Creates a fresh Q-table with a mild bias, same as the original scripts."""
    q = np.zeros((STATE_SPACE_SIZE, NUM_ACTIONS))
    for difficulty in range(NUM_DIFFICULTIES):
        for performance in range(NUM_PERFORMANCE_BUCKETS):
            for mastery in range(NUM_MASTERY_LEVELS):
                idx = get_state_index(difficulty, performance, mastery)
                if performance == 0:
                    q[idx][0] = 0.5  # bias towards decreasing difficulty
                elif performance == 2:
                    q[idx][2] = 0.5  # bias towards increasing difficulty
    return q


def get_state_index(difficulty, performance, mastery):
    return (
        difficulty * (NUM_PERFORMANCE_BUCKETS * NUM_MASTERY_LEVELS)
        + performance * NUM_MASTERY_LEVELS
        + mastery
    )


def get_performance_bucket(score):
    if score <= 3:
        return 0
    elif score <= 7:
        return 1
    return 2


def get_mastery_level(concepts_of_interest, concept_mastery):
    if not concepts_of_interest:
        return 1
    total_correct = sum(
        sum(concept_mastery.get(c, [])) for c in concepts_of_interest
    )
    total_questions = sum(
        len(concept_mastery.get(c, [])) for c in concepts_of_interest
    )
    if total_questions == 0:
        return 1
    accuracy = total_correct / total_questions
    return 0 if accuracy < 0.4 else 1 if accuracy < 0.7 else 2


def choose_action(q_table, state_index, epsilon=DEFAULT_EPSILON):
    """
    Real epsilon-greedy Q-learning action selection.

    The original scripts computed this Q-table update every run but then
    ignored it, using a hardcoded `if score < 5 / >= 8` rule instead — so
    the "RL" was cosmetic. This version actually uses the learned Q-values,
    falling back to random exploration with probability epsilon.
    """
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(NUM_ACTIONS))
    return int(np.argmax(q_table[state_index]))


def update_q_table(q_table, state_index, action, reward, next_state_index):
    old_value = q_table[state_index][action]
    next_max = np.max(q_table[next_state_index])
    q_table[state_index][action] = (1 - LEARNING_RATE) * old_value + LEARNING_RATE * (
        reward + DISCOUNT_FACTOR * next_max
    )


def calculate_reward(score, previous_score, difficulty):
    base_reward = 2 if score >= 8 else 1 if score >= 5 else 0 if score >= 3 else -2
    improvement_reward = 1 if score > previous_score else 0 if score == previous_score else -1
    difficulty_bonus = difficulty * 0.5 if score >= 5 else -difficulty * 0.5
    return base_reward + improvement_reward + difficulty_bonus


def adjust_difficulty(current_difficulty, action):
    if action == 0:
        return max(0, current_difficulty - 1)
    elif action == 2:
        return min(NUM_DIFFICULTIES - 1, current_difficulty + 1)
    return current_difficulty


# ---------------------------------------------------------------------------
# Question generation (now asks Gemini for JSON directly, no fragile string parsing)
# ---------------------------------------------------------------------------

_BACKUP_MCQ_OPTIONS = {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}


def _backup_questions():
    questions = [f"Backup question {i+1}" for i in range(10)]
    answers = ["Backup answer"] * 5 + [(_BACKUP_MCQ_OPTIONS, "A")] * 5
    concepts = ["general"] * 10
    is_mcq_flags = [False] * 5 + [True] * 5
    return questions, answers, concepts, is_mcq_flags


def generate_questions(model, difficulty, text):
    """
    Generates 5 open-ended + 5 MCQ questions grounded in `text`.

    Uses Gemini's structured JSON output mode instead of asking the model
    to follow a free-text template and then regex-parsing the result. This
    is far more robust: the previous approach silently fell back to
    placeholder "Backup question" text whenever the model's formatting
    drifted even slightly, with no warning to the caller.
    """
    difficulty_desc = DIFFICULTY_DESCRIPTIONS[DIFFICULTY_LEVELS[difficulty]]
    prompt = f"""
Generate 10 unique questions strictly based on the text below, at
{DIFFICULTY_LEVELS[difficulty]} difficulty ({difficulty_desc}).

Return exactly 5 open-ended questions and 5 multiple-choice questions.
All questions must be directly relevant to the text content.

Respond with ONLY a JSON array of 10 objects, no other text, matching this shape:
[
  {{
    "type": "open_ended",
    "concept": "concept_name",
    "question": "question text",
    "sample_answer": "concise sample answer"
  }},
  {{
    "type": "mcq",
    "concept": "concept_name",
    "question": "question text",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct": "A"
  }}
]

Text: {text}
"""
    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        items = json.loads(response.text)
        questions, answers, concepts, is_mcq_flags = [], [], [], []
        for item in items:
            questions.append(item["question"])
            concepts.append(item.get("concept", "general"))
            if item["type"] == "mcq":
                answers.append((item["options"], item["correct"]))
                is_mcq_flags.append(True)
            else:
                answers.append(item.get("sample_answer", "No answer provided"))
                is_mcq_flags.append(False)

        if len(questions) < 10:
            logger.warning(
                "Model returned %d questions instead of 10; padding with backups.",
                len(questions),
            )
            bq, ba, bc, bf = _backup_questions()
            needed = 10 - len(questions)
            questions += bq[:needed]
            answers += ba[:needed]
            concepts += bc[:needed]
            is_mcq_flags += bf[:needed]

        return questions[:10], answers[:10], concepts[:10], is_mcq_flags[:10]

    except Exception as e:
        logger.error("Error generating questions: %s", e)
        return _backup_questions()


# ---------------------------------------------------------------------------
# Answer evaluation
# ---------------------------------------------------------------------------

def _maybe_translate_to_english(user_answer):
    if _translator is None:
        return user_answer
    try:
        detected_lang = _translator.detect(user_answer).lang
        if detected_lang == "hi":
            return _translator.translate(user_answer, src="hi", dest="en").text
    except Exception as e:
        logger.warning("Translation error (%s), using original answer.", e)
    return user_answer


def evaluate_text_answer(model, question, user_answer, sample_answer, concept):
    user_answer = _maybe_translate_to_english(user_answer)

    prompt = f"""
Evaluate this user answer for similarity to the sample answer, based on the concept '{concept}'.

Score as:
- 1 if fully similar (nearly identical meaning)
- 0.5 if partially similar (some overlap in meaning)
- 0 if not similar (no meaningful match)

Respond with ONLY a JSON object: {{"score": <number>, "feedback": "<short text>"}}

Question: {question}
User Answer: {user_answer}
Sample Answer: {sample_answer}
"""
    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        data = json.loads(response.text)
        score = float(data.get("score", 0))
        feedback = str(data.get("feedback", "No feedback provided"))
        return score, feedback
    except Exception as e:
        logger.error("Error evaluating text answer: %s", e)
        return 0, "Evaluation failed"


def evaluate_mcq_answer(user_answer, correct_answer):
    return 1 if user_answer.upper() == correct_answer.upper() else 0