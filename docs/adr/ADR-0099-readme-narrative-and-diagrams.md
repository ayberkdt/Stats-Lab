# ADR-0099: README narrative, generated diagrams, and absolute links

- Status: Accepted
- Date: 2026-11-12

## Context

The README is the front page of the repository and, through
`readme = { file = "README.md" }` in `pyproject.toml`, the long description of
the package on PyPI. It had grown as a list of sections: it did not say who
the library is for or which problem it removes, the verification discipline
was spread over a "Design principles" list without numbers, the gallery
captions mixed claims with and without numbers, and every link and image was
relative. Relative paths resolve on GitHub but not on PyPI, where they point
to nothing.

Some README facts were already checked by
`tests/test_documentation_examples.py` (ADR-0095): every Python block runs and
its quoted values are compared; the package, public-name, equation, lab,
ATHENS-topic, and distribution-family counts are re-read; the version badge,
the optional extras, the `UQStudy` stage table, and the package table match
the code. Anything new that the README claims needs the same treatment.

## Decision

1. **Narrative.** The README opens with one claim (one probabilistic model
   answers forward, inverse, risk, and system-reliability questions, with
   every formula traced to a source), says who it is for and which problem it
   removes (scattered tools that each redefine the inputs, draw from their
   own seed, and report in their own format), and then follows the reader's
   tasks: Quick start; "Why trust it" (a table of verification counts and the
   test that enforces each); the question map ("One study, every question");
   systems engineering; a visual tour; the package map ("What's inside");
   a documentation map by intent (learning paths, foundation guides,
   mathematical and API references, the overview presentation in
   `docs/presentation/`, interactive explainers, labs and benchmarks, design
   records); installation with the `visualization`, `numerics`, and `dev`
   extras; citing; licence. Headings state a claim where the section has one
   (sunumatik's assertive-copy rule), tables carry the detail, and the Quick
   start code is unchanged so its quoted values stay checked.
2. **Counts that tests keep true.** Counts that grow with almost every change
   are written as floors (`2,200+ test cases`, `550+ primary sources`,
   `90+ decision records`, `50+ Markdown files`, `80+ property-based and
   metamorphic tests`). `ReadmeTests.assert_floor` requires each floor to be at
   most the repository count and at least 80% of it, so a floor stays true as
   the repository grows and is flagged before it becomes misleading. The test
   count uses the rule of `tools/build_presentation_data.py` (test methods of
   classes in `tests/test_*.py`). The figure text floor (9 pt, from
   `ANNOTATION_SIZE` in the theme), the contrast thresholds (4.5:1 and 3:1, from
   `tests/test_visualization_accessibility.py`), the CI Python range (from
   `.github/workflows/ci.yml`), the wheel steps (`twine check --strict`,
   `tools/smoke_test_wheel.py`), the zero-dependency badge, the BibTeX version,
   and the number of dependency layers (from `tests/test_architecture.py`) are
   read from their sources by the test. Existing exact counts are kept.
3. **Two generated diagrams.** `tools/build_readme_figures.py` draws, with the
   datawiz diagram toolkit (`datawiz.diyagram.Diyagram`, grid layout compiled
   by LaTeX), a workflow diagram (`JointInput` and a simulator into
   `ProbabilisticModel`, `UQStudy` and seven of its stages, one report, and the
   `SystemModel` to `MissionAssessment` branch) and a package-layer map read
   from the `LAYER` dictionary of `tests/test_architecture.py`, each in a light
   and a dark variant shown through `<picture>` and `prefers-color-scheme`.
   The build fails on any datawiz audit error. It also writes
   `docs/images/readme-diagrams.json` (stages, classes, and layers drawn);
   the README test compares that record with the architecture test's map,
   checks that every stage is a `UQStudy` method and every class is exported by
   its package, and checks that the README shows all four images. A layer
   change therefore fails the test until the diagram is rebuilt.
4. **Gallery.** The ten lab figures in `docs/images/` keep their file names and
   are ordered by the question map (forward, sensitivity, reliability, rare
   events, risk, extremes, calibration, model error, filtering, causal). Each
   caption is one sentence carrying the number shown in the figure.
5. **Absolute links instead of a generated PyPI readme.** Every link points to
   `https://github.com/ayberkdt/Stats-Lab/blob/main/...` (documents) or
   `/tree/main/...` (directories), and every image to
   `https://raw.githubusercontent.com/ayberkdt/Stats-Lab/main/...`. The
   alternative, a second readme rewritten at build time, needs either a
   committed generated file that can drift or a custom setuptools build step
   for a static `readme` field; both add a moving part to packaging for a
   problem a text convention solves. The cost of absolute links is that a
   released description points at `main`, not at the release tag, and that new
   images show only after they reach `main`. Two tests keep the convention:
   the README may contain no relative link or image, and every repository URL
   must map to a file (`blob`, raw) or directory (`tree`) that exists, with any
   heading anchor resolved.

### Task budget (written before the page)

```
GÖREV BÜTÇESİ — README.md                 (yüzey: web; GitHub and PyPI render the Markdown)
Birincil görev(ler)       : 1 decide whether the library fits · 2 install and run the Quick start · 3 find the right document
Birincil eylem            : copy the Quick start install command (no buttons)
Eylem / görünüm           : ≤ 5     · planned: 0 (links only)
Üst düzey gezinti         : ≤ 7     · planned: 0 (the host renders its own outline)
Seçenek kümesi            : ≤ 5 ya da varsayılan işaretli · planned: 4 install rows, "none" first
Form alanı / adım         : ≤ 7     · planned: 0
Geri bildirim             : static page, no waiting state
Stil bütçesi              : host stylesheet; render check ≤ 6 type sizes · 0 buttons · 1 link tone · 0 shadows · 1 radius
Sisteme yüklenen karmaşıklık : counts re-read by tests; figure theme chosen by prefers-color-scheme; absolute links, so PyPI needs no build step
```

### Figure briefs

```
FİGÜR BRİFİ — readme-one-model-workflow
Mesaj      : one model feeds every UQStudy stage and one report; systems feed the same report
Okur       : karma · Ortam: web (README, about 880 px wide)
Veri       : 13 nodes (2 inputs, model, study, 7 stages, report, 2 systems nodes), 19 directed edges
Biçim      : block/architecture diagram (diyagram.md "mimari şema"), grid placement, stages in one common region
Tema       : derin, light and dark (koyu=True)

FİGÜR BRİFİ — readme-package-layers
Mesaj      : 33 packages sit in 9 layers and imports point down only
Okur       : karma · Ortam: web
Veri       : LAYER of tests/test_architecture.py (root package left out: version metadata only)
Biçim      : layered block map, one row per layer (wrap after 6), layer label on the left; modeling filled
Tema       : derin, light and dark
```

## Consequences

- The README tells a newcomer in the first screen what the library is for;
  verification is stated as numbers with the test that keeps each true.
- Adding tests, sources, or decision records needs no README edit until a count
  exceeds its floor by 25%; then `assert_floor` names the floor to raise.
- Changing a layer in `tests/test_architecture.py`, a drawn stage, or a drawn
  class fails `test_readme_diagrams_match_the_code` until
  `PYTHONPATH=src python tools/build_readme_figures.py` is rerun. The build
  needs LaTeX and the datawiz toolkit, which are not library dependencies; the
  test needs neither.
- The gallery captions quote numbers read from the current figures. When the
  lab figures are regenerated, the captions must be re-read against the new
  PNGs (no test can read numbers from an image).
- Links in a released PyPI description point at `main`.
- Whether PyPI's sanitizer keeps `<picture>` and `<source>` was not checked
  here (the renderer was not installed). The `<img>` inside each `<picture>`
  carries the light image, so the figures still show if the wrappers are
  stripped; CI's `twine check --strict` validates the rendered description.

## Verification

- `tests/test_documentation_examples.py`: the README Python blocks run with
  their quoted values; `ReadmeTests` gains six tests
  (`test_verification_floors_match_the_repository`,
  `test_figure_ci_and_packaging_claims_match_their_sources`,
  `test_layer_count_matches_the_architecture_test`,
  `test_readme_diagrams_match_the_code`, `test_readme_uses_only_absolute_links`,
  `test_absolute_repository_urls_resolve`), and `number_before` accepts any
  white space between words. A mutation run (floor raised to 2,900, one image
  URL misspelt, one link made relative) failed four tests, each naming the
  defect.
- Diagram audit (`tools/build_readme_figures.py`, datawiz `latex_denetimi` on
  the compiled page): `readme-one-model-workflow` 0 HATA · 0 UYARI (209 mm),
  dark variant 0 HATA · 0 UYARI; `readme-package-layers` 0 HATA · 0 UYARI
  (190 mm), dark variant 0 HATA · 0 UYARI. `python -m datawiz.kod_denetimi
  tools/build_readme_figures.py`: 0 HATA · 0 UYARI. All four PNGs were
  inspected: no overlapping text, no line through text, every label readable
  in both themes. Two defects were found only by eye and fixed before
  delivery: grouped layer boxes left their titles on the neighbouring box
  border (replaced by row labels), and the subtitle said "dark block" for a
  block that is light blue in the dark variant (now "filled block").
- Gallery captions were checked against each PNG: 15x (0.0583 to 0.00388) at
  256 points; S = 0.01 and S_T = 0.25 for X3; beta = 1.42; four design points at
  beta = 3.00 and 4.0x; expected loss 3.53 and 99% CVaR 90.4; 40 maxima, 100-year
  level 20.5 with 15.4 to 47; 9 measurements up to t = 6 and bands to t = 12;
  45 of 45 against 35; 96% of steps; 3 of 4 intervals, 0.77.
- Interface gate: the README was rendered to HTML with `markdown-it-py`
  (CommonMark with tables, raw HTML kept, GitHub-like stylesheet with light and
  dark tokens) and audited with `ux-yasa-denetimi.mjs --tablo`: 0 HATA,
  1 UYARI. In a browser at 1280x800 the first screen holds the claim, the
  audience and problem paragraphs, and the top of the workflow diagram; at
  375 px the page has no horizontal scroll (tables and code scroll inside
  their boxes).

```
YASA DENETİMİ — README.md (rendered) · 2026-11-12 · denetleyici: koşuldu, 0 HATA · 1 UYARI

Sonuç: 20/20 yasa ölçüldü · 0 HATA · 3 UYARI (3 gerekçeli) · karar: TESLİM
```

| # | Yasa | Ölçüm | Eşik | Durum | Gerekçe (yalnız UYARI) |
|---|---|---|---|---|---|
| 1 | Hick | nav 0 · eylem 0 · select 0 | ≤ 7 · ≤ 5 · ≤ 7 | geçti | |
| 2 | Fitts | 0 buttons; targets are inline links in prose (line box 24 px), exempt as inline targets | ≥ 44 px · ≤ 24 px | geçti | |
| 3 | Jakob | conventional README order (title, badges, quick start, ..., licence); 0 scroll or cursor overrides | 0 sapma | geçti | |
| 4 | Seçim Yükü | install table 4 rows, "none" (no extra) first; radio 0 | ≤ 5 ya da varsayılan | geçti | |
| 5 | Yakınlık | paragraph gap 16 px / section gap 24 px plus a rule line = 1:1.5 | ≤ 1:2 | UYARI | Gaps come from the host stylesheet, which Markdown cannot change; each section also starts with a heading rule. |
| 6 | Benzerlik | 4 type sizes · 1 radius · 0 shadows · 0 button styles | ≤ 8 · ≤ 4 · ≤ 2 · ≤ 2 | geçti | |
| 7 | Tekdüze Bağlılık | 0 cards, 0 nested; 10 gallery figures in one 2-column table | ≤ 12 · 0 | geçti | |
| 8 | Ortak Bölge | single-item boxes 0; 1 top-level region (main) | 0 · ≤ 4 | geçti | |
| 9 | Prägnanz | glass 0 · glow 0 · gradient 0 | 0 · 0 · ≤ 3 | geçti | |
| 10 | Von Restorff | primary actions 0 · saturated hues 0; one emphasis per diagram (filled block) | 1 · ≤ 2 | geçti | |
| 11 | Miller | form 0 fields · tables at most 3 columns | ≤ 7 · ≤ 7 | geçti | |
| 12 | Sıra Konumu | first: claim and Quick start · last: Citing, Licence | birincil görevler | UYARI | A README ends with citing and licence by convention (Jakob); the primary tasks hold the first three sections. |
| 13 | Doherty | transitions 0 · animations 0; static page | ≤ 300 · 0 · < 400 ms | geçti | |
| 14 | Zeigarnik | 0 multi-step flows | ≥ 3 adımda gösterge | geçti | |
| 15 | Zirve-Son | static document: no success, empty, or error state exists | 3/3 ekran | kapsam dışı | |
| 16 | Parkinson | required inputs 0/0 | ≤ %50 · 0 | geçti | |
| 17 | Tesler | 0 inputs; counts re-read by tests, theme from prefers-color-scheme, absolute links need no PyPI build step | her girdiye gerekçe | geçti | |
| 18 | Postel | 0 inputs, 0 patterns | 0 · normalizasyon | geçti | |
| 19 | Occam | page 446 elements · main 440 | ≤ 900 · ≤ 150 | UYARI | The README is one long-form article on GitHub and PyPI; its ten headed sections act as the views, and every element maps to one of the three tasks. |
| 20 | Pareto | 1280x800 first screen: 721 of 747 content px (97%) serve task 1 (claim, audience, problem, solution, diagram); badges 26 px | ≥ %60 | geçti | |

Risks: the audit ran on a local rendering; GitHub and PyPI apply their own
stylesheets, so spacing and type sizes there may differ from rows 5 and 6.
The two new diagrams show on GitHub and PyPI only after this change reaches
`main` (their URLs point there).

References: [REF-PROBSTAT-REPORTING] (reporting contract), [REF-W3C-WCAG21-2018]
(contrast thresholds quoted in the README), [REF-PYTHON-AST] (the README
tests).
