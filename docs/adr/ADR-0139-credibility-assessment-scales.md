# ADR-0139 — Credibility assessment scales: NASA-STD-7009B, PCMM, and a generic scale

- Status: Accepted
- Date: 2026-11-24
- Extends: ADR-0048 (API stability)

## Context

Space agencies ask that the results of models and simulations used for
decisions be reported together with a structured credibility assessment.
`validation.credibility.CredibilityRecord` keeps verification, validation, and
prediction evidence apart and states its gaps, but it scores nothing against a
published scale. Teams that must report a NASA-STD-7009B assessment or a
Predictive Capability Maturity Model (PCMM) table keep it in a spreadsheet next
to the evidence, where the rules of the scale (no partial credit, no combining
of factors, caps between elements) are not enforced and the gap to a required
profile is recomputed by hand.

The risk of a scoring module is the opposite one: a roll-up that looks
authoritative and is not in the source. Both scales say so themselves.
NASA-STD-7009B states that the factors are largely independent and that
comparing or combining them should be avoided. The PCMM report calls its six
elements "apples and oranges" and recommends against aggregation.

## Decision

1. **A data-driven scale.** `CredibilityScale` holds named
   `CredibilityFactor` objects grouped by category, each with consecutive
   ordered `CredibilityLevel` objects (value 0, 1, ... and a one-line
   descriptor) and optional sub-factors. A scale also carries its source key,
   citation, the sections it was read from, a score resolution (`1.0` whole
   levels, `0.1` tenths, `0.0` any real), source-defined `LevelCap` rules
   between factors, and its roll-up rule. Any other scale can be built from
   the same objects, so a standard that could not be read is supplied by the
   user, never reconstructed from memory.
2. **Three shipped constructors**, each transcribed from the documents read
   (list below): `nasa_std_7009b_scale()`, `pcmm_scale()`, and
   `nasa_cas_2008_scale()`. Level descriptors are paraphrases of one line with
   a section or table citation recorded in `sections`, not the documents'
   wording.
3. **Roll-ups only where the source defines one.** The roll-up rule is
   `"none"` (NASA-STD-7009B: `overall` and `category_minimum` raise and say
   why), `"minimum"` (the 2008 scale: the overall score is the minimum of the
   eight factors, [EQ-VAL-CRED-001]) or `"triple"` (PCMM: minimum, mean, and
   maximum reported together, [EQ-VAL-CRED-002]). No mean is computed
   anywhere else. `weighted_evidence_review_level` implements the one weighted
   average a source defines ([EQ-VAL-CRED-004], review weight at most 0.3).
4. **Assessment as an explicit record.** `FactorScore` carries the factor,
   achieved level, evidence items, rationale, assessor and date (free text),
   and optional sub-factor levels. `CredibilityAssessment` requires every
   factor to be scored or marked not assessed with a reason; level 0
   ("insufficient evidence") is a score and `None` is its absence. Unknown
   factors, repeated factors, levels outside the range or off the scale's
   resolution, and violated source caps (PCMM: validation and uncertainty
   quantification at most two above the lower verification element,
   [EQ-VAL-CRED-003]) raise `ValueError`; a cap is enforced only when the
   limiting elements are assessed.
5. **Sufficiency is separate from assessment.** NASA-STD-7009B "levies no
   requirements with respect to what levels to achieve" and the PCMM report
   deliberately excludes adequacy from its scale, so no profile ships. A
   `TargetProfile` (required level per factor, the criticality it is for, its
   basis) is supplied by the caller. `level_gaps`, `is_sufficient`, and
   `improvement_plan` compute `max(0, required - achieved)`
   ([EQ-VAL-CRED-005]); a required factor with no assessment is a gap of
   unknown size, because absence of an assessment is not sufficiency. The
   improvement order (unassessed first, then decreasing gap, ties in scale
   order, plus the capping elements still below what the cap needs) is a
   convention of this library, labelled as such.
6. **Criticality matrix.** `criticality_assessment(consequence, influence)`
   and `criticality_matrix()` return the 5 by 5 red/yellow/green table of
   Appendix D, Figure 1 ([EQ-VAL-CRED-006]), checked against the rendered
   figure. The quantitative breakpoints between consequence levels are set
   per programme by the standard and are not encoded; the caller supplies the
   level numbers. The standard allows programmes to adjust the matrix, so the
   lookup is a default.
7. **Bridge, not automation.** `suggest_record_evidence` lists, per factor,
   which fields of a `CredibilityRecord` bear on it (numerical-error budgets
   for verification factors, validation assessments, prediction statements
   for uncertainty factors, domain checks for the use assessment). It returns
   descriptions for the assessor and never a level.
8. **Rendering and archiving.** `render_assessment`, `render_gaps`, and
   `render_improvement_plan` produce aligned text or Markdown tables;
   `to_dict` / `from_dict` on scale, assessment, and profile are
   JSON-compatible and validated again on load.
9. **Placement.** `validation/credibility_scale.py`, layer 5, imports only
   `foundations` and its sibling `validation.credibility`. It is added to the
   `validation` exports.

## Sources read, and what was not

Documents opened and read in this work (text extracted from the PDFs and the
cited parts read; level wording was paraphrased, never copied):

- NASA-STD-7009B, approved 2024-03-05. Read: 4.1.1.2 [M&S 6] and 4.1.1.5
  [M&S 43e] (criticality record and level thresholds), 4.2.1.8 [M&S 48]
  (capability factors), 4.3.3.2 [M&S 29] and 4.3.6 (results factors, [M&S 31],
  and the statement that no sufficiency levels are required), 4.3.8
  (reporting: best estimate, uncertainty statement [M&S 33], levels and gaps
  [M&S 50], [M&S 35], caveats [M&S 32]), Appendix D (criticality assessment:
  D.1, Figure 1, D.2, Table 3, D.3, Table 4) and Appendix E (E.1-E.2
  principles including the discrete-step and no-combining statements; E.3 with
  Tables 5-11; E.4 with Tables 12-20; E.5 reporting). Table 19 words the scope
  of a level-4 use/analysis review as "capability-related factors" where the
  level-3 row and E.4.6 speak of results factors; the descriptor repeats the
  table and says so.
- NASA-HDBK-7009B, approved 2026-02-03. Opened and text-extracted; Sections
  5.5-5.6 and Appendix D.2 were read for the statement that the capability and
  results assessments together convey the credibility of results and that
  credibility cannot be measured directly. No factor or level is taken from the
  handbook, and its other appendices (including the risk assessment of
  Appendix E) were not read in detail.
- M. Babula, W. J. Bertch, L. L. Green, J. P. Hale, G. E. Mosier, M. J. Steele,
  J. Woods, "NASA Standard for Models and Simulations: Credibility Assessment
  Scale", NTRS 20090005963. The task named this identifier for the earlier
  scale paper by Blattnig, Green, Luckring, Morrison, Tripathi, and Zang; the
  document behind the identifier is the Babula et al. paper, a secondary
  account of the 2008 standard. It was read in full (Section IV, Table 1,
  Roll-up Processes, Figures 1-2 as text). The Blattnig et al. paper (AIAA, NTRS 20080015742)
  was opened in the source verification pass: it proposes a different scale
  (seven categories, four levels numbered 1 to 4, required against achieved
  levels, and no roll-up: its authors do not favour averaging or taking the
  minimum across results), described there as sharing the structure of the
  appendix scale of the interim standard but differing in details, so it
  cannot verify the eight-factor scale taken from Babula et al., and nothing
  is attributed to it. The 2008 standard itself was not
  read, which is why the 2008 scale is marked historical and secondary.
- W. L. Oberkampf, M. Pilch, T. G. Trucano, "Predictive Capability Maturity
  Model for Computational Modeling and Simulation", SAND2007-5948 (OSTI).
  Read: Sections 3.5, 4.1 (six elements, scores in tenths), 4.2 and Table 3
  (descriptors), 4.2.5 (cap, dependency order), 5.1 (project maturity
  requirements), and 5.2 (aggregation, the worked example whose triple
  `(0.0, 1.1, 2.0)` is reproduced in the tests from the matrices of Eq. (2)).

## Consequences

- A NASA-STD-7009B assessment can be kept as data next to the evidence,
  rendered for the report, and compared with a programme's thresholds without
  a number the standard does not define.
- The assessment is a judgement on written descriptors. The module checks
  that the form is complete and consistent with the scale's rules; it cannot
  check that a level is deserved. Evidence items and rationales are free
  text, and the record of who judged and when is the assessor's.
- A PCMM or NASA score measures the rigour and maturity of the process
  evidence, not the accuracy of predictions; both documents say so and the
  scale `notes` repeat it.
- Descriptors are paraphrases; a decision that depends on the exact words of a
  level must read the standard. Revisions of a standard require a new
  constructor (the source key names the version), not an edit of the old one.
- The level counts and names are asserted in the tests with the section
  references in each test docstring, so a transcription error is a test
  failure to be compared with the document, not a silent drift.

## Deliberately left out

- The HDBK-7009B material beyond the cited sections, including its risk
  assessment, and any scale not opened here (other agency scales, the
  Harmon-Youngblood, Logan-Nitta, and Mehta scales named in the standard's
  references).
- The Evidence/Technical Review sub-factor weights of the 2008 scale as
  stored data: sub-factors can be scored and `weighted_evidence_review_level`
  combines them, but the ten weights of the paper's Figure 2 are not shipped.
- Quantitative consequence breakpoints and an automatic influence
  determination (Table 4); an automatic score from a `CredibilityRecord`;
  radar and bar charts of the levels (the standard's Appendix E.5 examples are
  figures; the tables here carry the same information).
- Built-in sufficiency profiles for a decision criticality. Neither source
  defines them; inventing a mapping from "red" to required levels would
  present a convention as a requirement.

## Verification

`tests/test_credibility_scale.py`: structure of each shipped scale against the
cited sections, level ordering and no partial credit, the PCMM triple against
the report's worked example, the PCMM cap, minimum roll-up and weighted score
on hand-built examples, the criticality matrix (rows, monotonicity, counts),
gap lists and improvement order, rejection of unknown factors and levels, the
record bridge, rendering, and the `to_dict` round trip through JSON.
