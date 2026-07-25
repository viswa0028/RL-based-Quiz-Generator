# -*- coding: utf-8 -*-
"""
cli_quiz.py

Terminal version of the Adaptive Quiz System. Generates 5 open-ended
and 5 multiple-choice questions from user-supplied text, scores the
attempt, and uses the shared RL policy in quiz_engine.py to pick the
next difficulty level.

Setup:
    pip install -r requirements.txt
    export GEMINI_API_KEY=your-key-here   # or put it in a .env file

Run:
    python cli_quiz.py
"""

from collections import defaultdict

from dotenv import load_dotenv

import quiz_engine as qe

load_dotenv()  # loads GEMINI_API_KEY from a .env file if present

concept_mastery = defaultdict(list)


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

    for concept, results in concept_results.items():
        concept_mastery[concept].extend(results)

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
    model = qe.setup_model()
    q_table = qe.new_q_table()

    current_difficulty = 1  # start at medium
    previous_score = 5
    epsilon = qe.DEFAULT_EPSILON

    print("Welcome to the Adaptive Quiz System!")
    print("Answer 5 text questions and 5 MCQs.")
    print("Please enter the text to generate questions from:")
    input_text = input()

    if not input_text.strip():
        print("Text cannot be empty. Please try again.")
        return

    questions, answers, concepts, is_mcq_flags = qe.generate_questions(
        model, current_difficulty, input_text
    )

    print(f"\nDifficulty: {qe.DIFFICULTY_LEVELS[current_difficulty]}")
    user_score, concept_results = take_quiz(model, questions, answers, concepts, is_mcq_flags)

    performance_bucket = qe.get_performance_bucket(previous_score)
    mastery_level = qe.get_mastery_level(list(concept_results.keys()), concept_mastery)
    current_state_index = qe.get_state_index(current_difficulty, performance_bucket, mastery_level)

    action = qe.choose_action(q_table, current_state_index, epsilon)
    next_difficulty = qe.adjust_difficulty(current_difficulty, action)

    reward = qe.calculate_reward(user_score, previous_score, current_difficulty)
    next_state_index = qe.get_state_index(
        next_difficulty,
        qe.get_performance_bucket(user_score),
        qe.get_mastery_level(list(concept_results.keys()), concept_mastery),
    )
    qe.update_q_table(q_table, current_state_index, action, reward, next_state_index)

    if user_score < 5:
        reason = "Score too low, moving to an easier level."
    elif user_score < 8:
        reason = "Good score, maintaining current level."
    else:
        reason = "Excellent score, increasing challenge."

    print(f"\nNext difficulty: {qe.DIFFICULTY_LEVELS[next_difficulty]} ({reason})")


if __name__ == "__main__":
    main()
