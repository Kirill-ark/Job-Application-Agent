# What failed and how I fixed it

A log of what the evals found, in order. The numbers in `results/latest.md`
are always the current ones; this file explains how they got there.

## Match scoring

**First run** (scoring as built in step 4):

| | all | tune | holdout |
|---|---|---|---|
| Accuracy | 66% | 72% | 56% |
| Balanced accuracy | 58% | 66% | 48% |

The first report listed misses from both splits. The fixes below were chosen
from the **tune** misses only, and each one is a general rule, not a tweak
for one vacancy.

1. **Non-tech jobs scored "maybe".** A marketing internship got 50: the ad
   names no skill from the vocabulary, and "no skills found" counted as a
   neutral 0.5. The vocabulary only has tech skills, so an ad that names
   none of them is almost never a tech job. Now it counts as 0.25.
2. **Internships in another city scored "bad".** A location mismatch counted
   as 0. That is right for a working student job, which runs alongside the
   studies, but people do move for a full-time internship of a few months.
   An internship elsewhere now counts as 0.5.
3. **The "maybe" threshold was too high.** On the tune split no "bad" pair
   scored between 25 and 45, while two "maybe" pairs scored 35 and 38. The
   threshold moved from 40 to 30. The "good" threshold (65) stayed: on the
   tune split any threshold leaves one miss there, so moving it would only fit
   a single pair.

**After the fixes:**

| | all | tune | holdout |
|---|---|---|---|
| Accuracy | 77% | 79% | 72% |
| Balanced accuracy | 72% | 77% | 68% |

The holdout gain (56% → 72%) without tuning on holdout suggests the rules
generalize rather than memorize the tune pairs.

**Still wrong, on purpose not tuned further:**

- Only German language requirements are understood. A Dutch-only internship
  ("Nederlands spreken een vereiste") now scores "maybe" instead of "bad".
  Fix later: detect other languages the CV does not have.
- An ad that names a single tech skill (a legal role mentioning "LLM") gets
  a high skill share from that one match. Smoothing helps but not enough.
- Two "good" pairs for the second CV score 64, one point under the threshold.
- Roles whose stack the CV lacks (JavaScript/PHP, Tableau) score "bad"
  where the rubric says "maybe". Arguably both answers are defensible.

## Cover letter groundedness

**First full run** (8 letters, 61 sentences, judge: Qwen 3.8 27B):
0 sentences unsupported, 10 (16%) "partly": embellishments the CV does not
back ("robust pipelines", "production-grade models", "strong competence",
"fluent" for C1), and one skill credited to the job instead of a project.
The rules caught none of them: they check skills and numbers, not wording.
The rules had one false alarm: a skill that was part of the vacancy title.

Also found: Groq allows Qwen only 1,000 output tokens per minute and refuses
requests whose max_tokens could exceed it, so the judge first failed on 5 of
8 letters. It now asks for short reasons with max_tokens 900.

**Fixes:**
1. The writer's prompt forbids judging words and asks for CV language levels.
2. The rules cut a named vacancy title out of a sentence before checking it.

**After the fixes** (8 new letters, 67 sentences):

| | first run | after fixes |
|---|---|---|
| Not backed by the CV (partly + unsupported) | 16% | 9% |
| Letters without any problem | 2 of 8 | 4 of 8 |
| False alarms of the rules | 1 | 0 |

I read the 6 remaining sentences myself:

- 4 are mild interpretations rather than facts ("deployment pipelines" for one
  Docker container, "production workflow", "demonstrating practical ML").
- 1 is a real gap in the rules: "My current studies in Computer Science at TU
  Berlin align with the role" is a claim the model marked as "other", so it
  cites nothing. It is true (fact ed1), but it skips the citation. The rules
  only catch such hidden claims when they contain a skill or a number.
  Fix later: an "other" sentence that names a CV employer, school or project
  is a claim.
- 1 is a judge error: a sentence about PyTorch cites the thesis (p1), which
  uses PyTorch, but the judge only looked at the job (e1). The judge is
  useful, not infallible, so its verdicts are read by a person too.
