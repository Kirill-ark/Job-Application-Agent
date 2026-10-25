# Eval results

Written by `python -m evals.run_evals`. Do not edit by hand.

## Match scoring

47 vacancy-CV pairs with hand labels (good / maybe / bad).
Holdout pairs were never used to adjust the scoring.

| | all | tune | holdout |
|---|---|---|---|
| Accuracy | 77% | 79% | 72% |
| Balanced accuracy | 72% | 77% | 68% |

Confusion matrix (rows: hand label, columns: system label):

| | good | maybe | bad | recall |
|---|---|---|---|---|
| **good** | 5 | 4 | 0 | 56% |
| **maybe** | 0 | 9 | 3 | 75% |
| **bad** | 0 | 4 | 22 | 85% |

### Misses

| Vacancy | CV | Hand | System | Score | Why the hand label |
|---|---|---|---|---|---|
| AI Engineer - Tech Stagaire | alex | bad | maybe | 50 | requires Dutch |
| Intern Technology Consultant - Data & AI | alex | bad | maybe | 45 | a consulting internship rather than engineering, and in Munich |
| Legal Counsel - IT & Data Protection | alex | bad | maybe | 44 | legal role, a different field |
| Legal Counsel - IT & Data Protection | jonas | bad | maybe | 44 | legal role, a different field |
| Working Student — AI Engineering & Document Intelligenc | jonas | good | maybe | 64 | AI working student role in Berlin; RAG chatbot with LangChain in the CV |
| Working Student NLP / Machine Learning | jonas | good | maybe | 64 | ML working student in Berlin; Python, PyTorch, FastAPI, Docker in the CV |
| Member of Technical Staff (MTS) Intern (m/f/d) | alex | good | maybe | 40 | engineering internship in Berlin at an AI-native company; English ad |
| Member of Technical Staff (MTS) Intern (m/f/d) | jonas | good | maybe | 40 | engineering internship in Berlin at an AI-native company |
| Junior Data Analyst (m/w/d) | alex | maybe | bad | 29 | data role in Berlin, but a full-time junior position and Tableau is missing |
| Junior Data Analyst (m/w/d) | jonas | maybe | bad | 29 | data role in Berlin, but a full-time junior position and Tableau is missing |
| Praktikum Anwendungsentwicklung (w/m/d) | jonas | maybe | bad | 19 | development internship in Berlin, German is fine, but the stack (JavaScript, PHP) is not in the CV |

_Ran at 2026-10-25T20:43:31+00:00._
