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

## Cover letter groundedness

8 letters, 67 sentences (37 claims about the applicant). Each sentence was checked by the rules and by an LLM judge of another model family than the writer.

| | sentences | share |
|---|---|---|
| Judge: supported | 61 | 91% |
| Judge: wrong citation | 0 | 0% |
| Judge: partly | 5 | 7% |
| Judge: unsupported | 1 | 1% |

**Hallucination rate: 9% of sentences** (judged partly or unsupported). Letters without any: 4 of 8.

Rules compared with the judge:

- flagged by the rules: 0, of which the judge also found a problem in 0
- false alarms (rules flagged, judge says supported): 0
- missed by the rules (judge found a problem): 6

### Sentences the judge found not backed by the CV

| Letter | Sentence | Judge |
|---|---|---|
| Internship - Machine Learning Research Engine (jonas) | I am excited to apply for the Machine Learning Research Engineer internship because my academic work and practical experience have centered on machine learning, deep learning, and PyTorch, which align with the role’s focus. | e1 uses XGBoost, not deep learning or PyTorch. |
| Working Student — AI Engineering & Document I (alex) | I also served a machine‑learning model through a FastAPI endpoint in a Docker container, giving me experience with APIs and deployment pipelines. | p2.f2 matches; 'deployment pipelines' overstates a single Docker container. |
| Working Student — AI Engineering & Document I (jonas) | My current studies in Computer Science at TU Berlin align with the role’s focus on AI and document processing. | True per ed1, but no facts cited. |
| Working Student — AI Engineering & Document I (jonas) | In a recent project I deployed a RAG chatbot in Docker, gaining experience with containerised AI services. | Adds interpretation not in cited fact. |
| Working Student — AI Engineering & Document I (jonas) | My bachelor thesis achieved a 0.74 IoU on satellite image segmentation using PyTorch, demonstrating practical machine‑learning implementation. | Adds interpretation not in cited fact. |
| Member of Technical Staff (MTS) Intern (m/f/d (jonas) | In the same role I created data pipelines with Airflow and PostgreSQL to support the model’s production workflow. | Adds 'production workflow' not stated in e1.f2. |

_Ran at 2026-10-25T21:14:04+00:00._

## Tool use

4 full agent runs on the frozen vacancies. Finished: 3 of 4. **Without any tool-use mistake: 3 of 4.** Mean 11,762 tokens and 79 s per run.

| Goal | Status | Letters | Tool errors | Mistakes |
|---|---|---|---|---|
| default goal | done | 3 | 0 | none |
| at most two letters | failed: Stopped after 3 steps in a row in which every tool call failed | 0 | 24 | made-up ids: j1, j2, j3, j4, j5, j6, j7, j8; 16 repeated scores |
| no letters | done | 0 | 0 | none |
| goal in German | done | 3 | 0 | none |

_Ran at 2026-10-26T12:23:47+00:00._
