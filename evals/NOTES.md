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
