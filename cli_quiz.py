# -*- coding: utf-8 -*-
"""
cli_quiz.py

Terminal version of the Adaptive Quiz System.

Now persists the Q-table, concept mastery, and current difficulty to a
local JSON file between runs (profiles/<name>.json), so repeated practice
sessions actually accumulate learning instead of resetting every launch.
No accounts needed here since it's a single local user — pass --profile
to keep separate save files for different people sharing one machine.

Setup:
    pip install -r requirements.txt
    export GEMINI_API_KEY=your-key-here   # or put it in a .env file

Run:
    python cli_quiz.py                  # uses profiles/default.json
    python cli_quiz.py --profile alice  # uses profiles/alice.json
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv

import quiz_engine as qe

load_dotenv()

PROFILES_DIR = "profiles"


def profile_path(name):
    return os.path.join(PROFILES_DIR, f"{name}.json")


def load_profile(name):
    path = profile_path(name)
    if not os.path.exists(path):
        return {
            "q_table": qe.new_q_table(),
            "concept_mastery": defaultdict(list),
            "current_difficulty": 1,
            "previous_score": 5,
        }
    with open(path, "r") as f:
        data = json.load(f)
    return {
        "q_table": np.array(data["q_table"]),
        "concept_mastery": defaultdict(list, data["concept_mastery"]),
        "current_difficulty": data["current_difficulty"],
        "previous_score": data["previous_score"],
    }


def save_profile(name, q_table, concept_mastery, current_difficulty, previous_score):
    os.makedirs(PROFILES_DIR, exist_ok=True)
    with open(profile_path(name), "w") as f:
        json.dump(
            {
                "q_table": q_table.tolist(),
                "concept_mastery": dict(concept_mastery),
                "current_difficulty": current_difficulty,
                "previous_score": previous_score,
            },
            f,
            indent=2,
        )


def take_quiz(model, questions, answers, concepts, is_mcq_flags):
    print("\n--- Quiz ---")
    print("Answer in English or Hindi for text questions; A/B/C/D for MCQs:")

    quiz_data = []
    for i, (question, answer_data, concept, is_mcq) in enumerate(
        zip(questions, answers, concepts, is_mcq_flags)
    ):
        print(f"\nQ{i + 1}. [{concept}] {question}")
        if is_mcq:
            options, _ = answer_data
            for opt, text in options.items():
                print(f"{opt}. {text}")
            user_answer = input("Your answer (A/B/C/D): ").strip().upper()
            while user_answer not in ["A", "B", "C", "D"]:
                print("Invalid input, enter A, B, C, or D:")
                user_answer = input("Your answer (A/B/C/D): ").strip().upper()
        else:
            user_answer = input("Your answer: ").strip()
        quiz_data.append((question, user_answer, answer_data, concept, is_mcq))

    total_score = 0
    concept_results = defaultdict(list)
    weak_areas = defaultdict(list)

    for question, user_answer, answer_data, concept, is_mcq in quiz_data:
        if is_mcq:
            _, correct_answer = answer_data
            score = qe.evaluate_mcq_answer(user_answer, correct_answer)
            feedback = "Correct" if score == 1 else f"Incorrect (correct answer: {correct_answer})"
        else:
            sample_answer = answer_data
            score, feedback = qe.evaluate_text_answer(
                model, question, user_answer, sample_answer, concept
            )

        concept_results[concept].append(score)
        total_score += score
        threshold = 1 if is_mcq else 0.5
        if score < threshold:
            weak_areas[concept].append(feedback)

    feedback_para = "Review: "
    if weak_areas:
        feedback_para += "You need improvement in " + ", ".join(
            f"'{c}'" for c in weak_areas.keys()
        ) + "."
    else:
        feedback_para += "You performed well across all concepts!"

    print("\n" + feedback_para)
    print(f"Score: {total_score}/10")
    return total_score, concept_results


def main():
    parser = argparse.ArgumentParser(description="Adaptive Quiz System (CLI)")
    parser.add_argument(
        "--profile", default="default", help="Save file name under profiles/ (default: 'default')"
    )
    args = parser.parse_args()

    profile = load_profile(args.profile)
    q_table = profile["q_table"]
    concept_mastery = profile["concept_mastery"]
    current_difficulty = profile["current_difficulty"]
    previous_score = profile["previous_score"]

    model = qe.setup_model()

    print("Welcome to the Adaptive Quiz System!")
    print(f"Profile: {args.profile} | Current difficulty: {qe.DIFFICULTY_LEVELS[current_difficulty]}")
    print("Answer 5 text questions and 5 MCQs.")
    print("Please enter the text to generate questions from:")
    input_text = input()

    if not input_text.strip():
        print("Text cannot be empty. Please try again.")
        return

    questions, answers, concepts, is_mcq_flags = qe.generate_questions(
        model, current_difficulty, input_text
    )

    user_score, concept_results = take_quiz(model, questions, answers, concepts, is_mcq_flags)

    for concept, results in concept_results.items():
        concept_mastery[concept].extend(results)

    performance_bucket = qe.get_performance_bucket(previous_score)
    mastery_level = qe.get_mastery_level(list(concept_results.keys()), concept_mastery)
    current_state_index = qe.get_state_index(current_difficulty, performance_bucket, mastery_level)

    action = qe.choose_action(q_table, current_state_index)
    next_difficulty = qe.adjust_difficulty(current_difficulty, action)

    reward = qe.calculate_reward(user_score, previous_score, current_difficulty)
    next_state_index = qe.get_state_index(
        next_difficulty,
        qe.get_performance_bucket(user_score),
        qe.get_mastery_level(list(concept_results.keys()), concept_mastery),
    )
    qe.update_q_table(q_table, current_state_index, action, reward, next_state_index)

    save_profile(args.profile, q_table, concept_mastery, next_difficulty, user_score)

    if user_score < 5:
        reason = "Score too low, moving to an easier level."
    elif user_score < 8:
        reason = "Good score, maintaining current level."
    else:
        reason = "Excellent score, increasing challenge."

    print(f"\nNext difficulty: {qe.DIFFICULTY_LEVELS[next_difficulty]} ({reason})")
    print(f"Progress saved to {profile_path(args.profile)}")


if __name__ == "__main__":
    main()