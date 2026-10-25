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

8 letters, 61 sentences (35 claims about the applicant). Each sentence was checked by the rules and by an LLM judge of another model family than the writer.

| | sentences | share |
|---|---|---|
| Judge: supported | 51 | 84% |
| Judge: wrong citation | 0 | 0% |
| Judge: partly | 10 | 16% |
| Judge: unsupported | 0 | 0% |

**Hallucination rate: 16% of sentences** (judged partly or unsupported). Letters without any: 2 of 8.

Rules compared with the judge:

- flagged by the rules: 1, of which the judge also found a problem in 0
- false alarms (rules flagged, judge says supported): 1
- missed by the rules (judge found a problem): 10

### Sentences the judge found not backed by the CV

| Letter | Sentence | Judge |
|---|---|---|
| Internship - Machine Learning Research Engine (jonas) | I also built robust data pipelines with Airflow and PostgreSQL. | 'Robust' is an adjective not in the CV. |
| Working Student — AI Engineering & Document I (alex) | I am excited to apply for the Working Student – AI Engineering & Document Intelligence role because it aligns perfectly with my focus on machine learning and backend development. | Backend development not shown by p1 or p3. |
| Working Student — AI Engineering & Document I (alex) | My current studies in Computer Science and hands‑on Python development equip me to contribute to document parsing and AI model prototyping. | Backend development not supported by cited facts. |
| Working Student — AI Engineering & Document I (jonas) | I am fluent in English and German and can work up to 20 hours per week, fitting the part‑time requirements of the role. | C1 is not 'fluent'; overstates English level. |
| Member of Technical Staff (MTS) Intern (m/f/d (jonas) | My current role as a Working Student in Machine Learning at Muster Logistics GmbH has given me hands‑on experience developing production‑grade models and data pipelines. | "Production-grade" overstates; CV doesn't claim production deployment. |
| Member of Technical Staff (MTS) Intern (m/f/d (jonas) | I built a delivery time forecasting model with Python and XGBoost, improving prediction accuracy, and I constructed robust Airflow pipelines that integrate with PostgreSQL databases. | "Robust" is an adjective not supported by any CV fact. |
| Member of Technical Staff (MTS) Intern (m/f/d (jonas) | I also developed a Retrieval‑Augmented Generation chatbot for study regulations, containerised it with Docker, and deployed the service for continuous use. | "For continuous use" overstates; CV only says deployed in Docker. |
| Working Student NLP / Machine Learning (alex) | In my current role I automated a weekly sales report, gaining practical experience with Python scripting and Dockerized services. | Dockerized services not in e1; Docker is from projects, not this role. |
| Working Student NLP / Machine Learning (jonas) | During my working‑student role at Muster Logistics GmbH I developed a delivery‑time forecasting model with Python and XGBoost, reducing error by 18 % and built robust data pipelines with Airflow and PostgreSQL. | "Robust" is an adjective not supported by CV. |
| Working Student NLP / Machine Learning (jonas) | In my bachelor thesis I implemented an image‑segmentation model using PyTorch (U‑Net) that achieved a 0.74 IoU on the test set, demonstrating strong competence with deep‑learning frameworks. | "Strong competence" overstates what the fact shows. |

### False alarms of the rules

| Letter | Sentence | Rules |
|---|---|---|
| Working Student NLP / Machine Learning (jonas) | I am excited to apply for the Working Student NLP / Machine Learning position because it aligns with my hands‑on experience in Python, PyTorch, Docker and FastAPI. | mentions skills the cited facts do not show: NLP |

_Ran at 2026-10-25T21:02:55+00:00._
