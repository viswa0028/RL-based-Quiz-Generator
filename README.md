# Adaptive Quiz Generator

A quiz system that generates questions from any text you provide, scores
your answers with an LLM, and adjusts difficulty over time using a
Q-learning policy that actually persists across sessions. Available as a
terminal app and a Flask web app with accounts and a progress dashboard.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [How It Works](#how-it-works)
6. [Contributing](#contributing)
7. [License](#license)

## Project Overview

Given a block of text, the system generates 5 open-ended questions and 5
multiple-choice questions grounded in that text using Gemini, grades your
answers, tracks your accuracy per concept, and picks the next difficulty
level (easy / medium / hard) using an epsilon-greedy Q-learning policy
over `(difficulty, recent performance, concept mastery)` state — with
that state actually saved between runs.

## Features

- **RL-based difficulty selection** — an actual epsilon-greedy Q-learning
  policy over difficulty/performance/mastery state, not a hardcoded
  if/else (see [How It Works](#how-it-works)).
- **Persistent learning** — the Q-table and concept mastery are saved
  between sessions (SQLite for the web app, a local JSON profile for the
  CLI), so the policy accumulates learning over time instead of resetting
  every run.
- **User accounts (web app)** — register/log in, and view a dashboard of
  quiz history, average score, and strongest/weakest concepts.
- **Structured question generation** — Gemini is asked to return JSON
  directly, so parsing is robust instead of regex-matching free text.
- **Per-concept mastery tracking** — weak concepts are surfaced in your
  feedback after each quiz.
- **Optional Hindi input** — text answers can be submitted in Hindi and
  are translated before grading.
- **Two front ends** — `cli_quiz.py` for the terminal (local JSON
  profiles, supports multiple `--profile` users on one machine),
  `web_quiz.py` for a browser-based version with real accounts and a
  SQLite-backed dashboard.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/viswa0028/RL-based-Quiz-Generator.git
   cd RL-based-Quiz-Generator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API key — copy `.env.example` to `.env` and fill in your
   [Gemini API key](https://aistudio.google.com/apikey):
   ```
   cp .env.example .env
   ```
   **Never commit your `.env` file or hardcode the key in source.**

## Usage

**Terminal version:**
```
python cli_quiz.py                  # uses profiles/default.json
python cli_quiz.py --profile alice  # separate save file per person
```
```
Please enter the text to generate questions from:
> Photosynthesis is the process by which plants convert light into energy...
Difficulty: medium
Q1. [photosynthesis] What raw materials do plants use during photosynthesis?
...
Next difficulty: hard (Excellent score, increasing challenge.)
Progress saved to profiles/alice.json
```

**Web version:**
```
python web_quiz.py
```
Then open `http://127.0.0.1:5000`, register an account, and take a quiz.
Visit `/dashboard` any time to see your quiz history and per-concept
accuracy. Your Q-table and difficulty carry over the next time you log
in — the app creates `quiz_app.db` (SQLite) on first run.

## How It Works

### Question Generation
`quiz_engine.generate_questions()` prompts Gemini for a JSON array of 10
questions (5 open-ended, 5 MCQ) grounded in the supplied text, each tagged
with a concept label. Falling back to placeholder questions only happens
if the model output can't be parsed at all, and is logged rather than
silent.

### Answer Evaluation
- MCQs are scored by exact match.
- Open-ended answers are scored 0 / 0.5 / 1 by asking Gemini to judge
  similarity to a sample answer, with brief feedback returned alongside
  the score.
- Text answers detected as Hindi are translated to English before grading.

### Difficulty Adaptation (Q-learning)
State = `(current_difficulty, performance_bucket, concept_mastery_level)`.
After each quiz:
1. A reward is computed from the score, improvement over the previous
   attempt, and current difficulty.
2. `choose_action()` picks decrease/keep/increase difficulty using
   epsilon-greedy selection over the Q-table — mostly exploiting the
   best-known action, occasionally exploring at random.
3. The Q-table is updated with the standard Q-learning update rule.

### Persistence
- **Web app** (`db.py` + SQLite): each user's Q-table, concept mastery,
  current difficulty, and full quiz-attempt history are stored server-side
  and reloaded on login, so learning compounds across sessions. The
  `/dashboard` route aggregates attempt history into average score and
  best/worst concept.
- **CLI** (`cli_quiz.py`): the same state is saved to
  `profiles/<name>.json` after every quiz and reloaded on the next run.
  Use `--profile` to keep separate histories for different people sharing
  one machine.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.