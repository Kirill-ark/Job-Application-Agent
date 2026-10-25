# How vacancy-CV pairs are labeled

Each pair gets one label: what a careful career adviser would tell this
student about applying. Labels were drafted by Claude **without looking at
the system's scores**, then checked by the project owner.

Both CVs are students in Berlin looking for working student (Werkstudent)
or internship roles.

## good
All of these hold:
- the role is in the CV's field (ML, data, AI, backend with Python);
- the level fits: working student or internship;
- it is in Berlin or remote (from Germany);
- the language requirement is met by the CV;
- the CV shows most of the core tools the ad asks for.

## maybe
The field fits, but exactly one notable drawback:
- an internship in another city (an internship can be worth moving for);
- an entry/junior or unclear level that might still take a student;
- the ad wants more German than the CV has (one level short);
- several core tools of the ad are missing from the CV.

## bad
Any of these:
- a different field (sales, legal, marketing, property management, IT support, ...);
- senior, lead or clearly experienced roles, also when the ad does not say
  "senior" but asks the person to own production systems;
- a working student role in another city that is not remote
  (it runs alongside the studies, so moving is not an option);
- a language the CV does not have (Dutch, French), or German two levels short;
- two or more "maybe" drawbacks at once.

Vacancies whose id starts with `synthetic-` were written by hand to cover
cases the live job board did not have (for example a German-only working
student role in Berlin). Everything else is a real Arbeitnow vacancy,
frozen on 2026-10-25 so the evals give the same input every run.
