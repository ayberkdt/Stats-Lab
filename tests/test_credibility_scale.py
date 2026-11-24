"""Credibility assessment scales: structure, roll-ups, gaps, and archiving.

The shipped scales are checked against the sections of the documents they
were transcribed from (NASA-STD-7009B Appendices D and E, the PCMM report,
the 2008 scale paper): factor counts and names, level counts, the roll-up
each source defines, and the criticality matrix, which was also compared with
the rendered Figure 1 of the standard. The PCMM aggregation triple is checked
against the worked example of the report (Section 5.2, Eqs. (2)-(3)). Gap
analysis, the improvement order, rejection of unknown factors and levels, the
bridge from a credibility record, rendering, and the round trip through
``to_dict`` are checked on hand-built examples. The tests never open the
source documents.

References:
    [REF-NASA-STD-7009B-2024], [REF-OBERKAMPF-PILCH-TRUCANO-2007],
    [REF-BABULA-ETAL-2009]. Checks [EQ-VAL-CRED-001], [EQ-VAL-CRED-002],
    [EQ-VAL-CRED-003], [EQ-VAL-CRED-004], [EQ-VAL-CRED-005], [EQ-VAL-CRED-006].

"""

from __future__ import annotations

import json
import re
import unittest
from collections.abc import Mapping
from pathlib import Path

from probstat.uq import NumericalErrorBudget
from probstat.validation import (
    CredibilityAssessment,
    CredibilityFactor,
    CredibilityLevel,
    CredibilityRecord,
    CredibilityScale,
    FactorScore,
    LevelCap,
    PredictionStatement,
    TargetProfile,
    VerificationEvidence,
    criticality_assessment,
    criticality_matrix,
    improvement_plan,
    is_sufficient,
    level_gaps,
    nasa_cas_2008_scale,
    nasa_std_7009b_scale,
    pcmm_aggregate_triple,
    pcmm_scale,
    render_assessment,
    render_gaps,
    render_improvement_plan,
    suggest_record_evidence,
    uniform_profile,
    weighted_evidence_review_level,
)

ROOT = Path(__file__).resolve().parents[1]

CAPABILITY = [
    "M&S Data Pedigree",
    "M&S Verification",
    "M&S Validation",
    "M&S Development Technical Review",
    "M&S Development Process/Product Management",
]
RESULTS = [
    "M&S Use Assessment",
    "M&S Input Pedigree",
    "M&S Uncertainty Characterization",
    "M&S Results Robustness",
    "M&S Use/Analysis Technical Review",
    "M&S Use Process/Product Management",
]
PCMM_ELEMENTS = [
    "Representation and Geometric Fidelity",
    "Physics and Material Model Fidelity",
    "Code Verification",
    "Solution Verification",
    "Model Validation",
    "Uncertainty Quantification and Sensitivity Analysis",
]
CAS_2008 = [
    "Verification",
    "Validation",
    "Input Pedigree",
    "Results Uncertainty",
    "Results Robustness",
    "Use History",
    "M&S Management",
    "People Qualifications",
]


def _scores(scale: CredibilityScale, levels: Mapping[str, float | None]) -> tuple:
    """Build one score per factor; factors absent from ``levels`` get level 0.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    out = []
    for name in scale.factor_names:
        level = levels.get(name, 0)
        reason = "" if level is not None else "not examined in this exercise"
        out.append(FactorScore(name, level, ("test evidence",), reason or "test"))
    return tuple(out)


def _assess(
    scale: CredibilityScale, levels: Mapping[str, float | None]
) -> CredibilityAssessment:
    """Build an assessment from a level mapping.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    return CredibilityAssessment(scale, _scores(scale, levels), subject="test model")


class Nasa7009BStructureTests(unittest.TestCase):
    """The 7009B scale has the factors, grouping and levels of Appendix E.

    References:
        [REF-NASA-STD-7009B-2024, Appendix E]. Checks [EQ-VAL-CRED-001].

    """

    def test_capability_factors_follow_e31_and_requirement_m_and_s_48(self) -> None:
        """Section E.3.1 and [M&S 48] (4.2.1.8) list five capability factors.

        References:
            [REF-NASA-STD-7009B-2024, Section 4.2.1.8],
            [REF-NASA-STD-7009B-2024, Appendix E.3.1].

        """
        scale = nasa_std_7009b_scale()
        category = scale.categories[0]
        self.assertEqual([f.name for f in scale.factors_in(category)], CAPABILITY)

    def test_results_factors_follow_e41_and_requirement_m_and_s_31(self) -> None:
        """Section E.4.1 and [M&S 31] (4.3.6.1) list six results factors.

        References:
            [REF-NASA-STD-7009B-2024, Section 4.3.6.1],
            [REF-NASA-STD-7009B-2024, Appendix E.4.1].

        """
        scale = nasa_std_7009b_scale()
        category = scale.categories[1]
        self.assertEqual([f.name for f in scale.factors_in(category)], RESULTS)
        self.assertEqual(len(scale.factors), 11)
        self.assertEqual(len(scale.categories), 2)

    def test_every_factor_has_five_ordered_levels_zero_to_four(self) -> None:
        """Tables 7-11 and 15-20 give levels 0-4, level 0 being no evidence.

        References:
            [REF-NASA-STD-7009B-2024, Appendix E.1].

        """
        scale = nasa_std_7009b_scale()
        for factor in scale.factors:
            self.assertEqual([lv.value for lv in factor.levels], [0, 1, 2, 3, 4])
            self.assertEqual(factor.levels[0].descriptor, "Insufficient evidence.")
            self.assertEqual(factor.max_level, 4)
            texts = [lv.descriptor for lv in factor.levels]
            self.assertEqual(len(set(texts)), 5, factor.name)

    def test_source_key_sections_and_rollup(self) -> None:
        """The scale cites its source and defines no roll-up (E.2).

        References:
            [REF-NASA-STD-7009B-2024, Appendix E.2].

        """
        scale = nasa_std_7009b_scale()
        self.assertEqual(scale.source_key, "REF-NASA-STD-7009B-2024")
        self.assertEqual(scale.rollup, "none")
        self.assertEqual(scale.resolution, 1.0)
        self.assertTrue(any("Tables 7-11" in s for s in scale.sections))
        self.assertTrue(any("Tables 15-20" in s for s in scale.sections))

    def test_whole_levels_only_no_partial_credit(self) -> None:
        """E.3.1 says each level is a discrete step with no partial credit.

        References:
            [REF-NASA-STD-7009B-2024, Appendix E.3.1].

        """
        scale = nasa_std_7009b_scale()
        with self.assertRaises(ValueError):
            _assess(scale, {"M&S Verification": 2.5})
        with self.assertRaises(ValueError):
            _assess(scale, {"M&S Verification": 5})
        with self.assertRaises(ValueError):
            _assess(scale, {"M&S Verification": -1})
        _assess(scale, {"M&S Verification": 4})

    def test_no_overall_and_no_category_minimum(self) -> None:
        """The standard says factor assessments should not be combined (E.2).

        References:
            [REF-NASA-STD-7009B-2024, Appendix E.2].

        """
        assessment = _assess(nasa_std_7009b_scale(), {"M&S Verification": 3})
        with self.assertRaises(ValueError) as caught:
            assessment.overall()
        self.assertIn("should not be combined", str(caught.exception))
        with self.assertRaises(ValueError):
            assessment.category_minimum()
        text = render_assessment(assessment)
        self.assertIn("reported separately", text)
        self.assertNotIn("Overall", text)

    def test_every_factor_must_be_scored_or_marked_not_assessed(self) -> None:
        """[M&S 48] and [M&S 31] require every factor to be recorded.

        References:
            [REF-NASA-STD-7009B-2024, Section 4.3.6.1].

        """
        scale = nasa_std_7009b_scale()
        scores = list(_scores(scale, {}))
        with self.assertRaises(ValueError) as caught:
            CredibilityAssessment(scale, tuple(scores[:-1]))
        self.assertIn("missing", str(caught.exception))
        assessment = _assess(scale, {"M&S Results Robustness": None})
        self.assertEqual(assessment.unassessed, ("M&S Results Robustness",))
        self.assertFalse(assessment.is_complete)
        self.assertIsNone(assessment.level("M&S Results Robustness"))


class PcmmStructureTests(unittest.TestCase):
    """The PCMM scale has six elements on levels 0-3 with tenths and a cap.

    References:
        [REF-OBERKAMPF-PILCH-TRUCANO-2007, Table 1]. Checks [EQ-VAL-CRED-002],
        [EQ-VAL-CRED-003].

    """

    def test_six_elements_four_levels_table_one(self) -> None:
        """Table 1 has six element rows and maturity levels 0 to 3.

        References:
            [REF-OBERKAMPF-PILCH-TRUCANO-2007, Table 1].

        """
        scale = pcmm_scale()
        self.assertEqual(list(scale.factor_names), PCMM_ELEMENTS)
        for factor in scale.factors:
            self.assertEqual([lv.value for lv in factor.levels], [0, 1, 2, 3])
        self.assertEqual(scale.source_key, "REF-OBERKAMPF-PILCH-TRUCANO-2007")
        self.assertEqual(scale.rollup, "triple")
        self.assertEqual(scale.resolution, 0.1)

    def test_aggregation_example_of_section_five_two(self) -> None:
        """Eqs. (2)-(3): four subsystems of six elements give (0.0, 1.1, 2.0).

        References:
            [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 5.2, Eq. (3)].
            Checks [EQ-VAL-CRED-002].

        """
        subsystems = (
            (1, 1.5, 1, 0, 0.5, 1),
            (1.5, 1, 0, 0.5, 1.5, 0),
            (2, 1.5, 0.5, 1, 1.5, 1),
            (2, 2, 1, 0.5, 1.5, 1.5),
        )
        flat = [score for row in subsystems for score in row]
        low, mean, high = pcmm_aggregate_triple(flat)
        self.assertEqual(low, 0.0)
        self.assertEqual(high, 2.0)
        self.assertAlmostEqual(mean, 25.5 / 24.0, places=14)
        self.assertEqual(round(mean, 1), 1.1)
        with self.assertRaises(ValueError):
            pcmm_aggregate_triple([])
        with self.assertRaises(ValueError):
            pcmm_aggregate_triple([1.0, float("nan")])

    def test_assessment_overall_is_the_triple_not_a_mean(self) -> None:
        """The overall of a triple scale carries minimum, mean, and maximum.

        References:
            [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 5.2].
            Checks [EQ-VAL-CRED-002].

        """
        levels = dict(zip(PCMM_ELEMENTS, (1, 1.5, 1, 0, 0.5, 1), strict=True))
        overall = _assess(pcmm_scale(), levels).overall()
        self.assertEqual(overall.rule, "triple")
        self.assertEqual(overall.minimum, 0.0)
        self.assertEqual(overall.maximum, 1.5)
        self.assertAlmostEqual(overall.mean or 0.0, 5.0 / 6.0, places=14)
        self.assertEqual(overall.weakest, ("Solution Verification",))

    def test_tenths_allowed_other_fractions_rejected(self) -> None:
        """Section 4.1 allows scores in tenths such as 1.5, not 1.55.

        References:
            [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 4.1].

        """
        scale = pcmm_scale()
        assessment = _assess(scale, {"Code Verification": 1.5})
        self.assertEqual(assessment.level("Code Verification"), 1.5)
        with self.assertRaises(ValueError):
            _assess(scale, {"Code Verification": 1.55})
        with self.assertRaises(ValueError):
            _assess(scale, {"Code Verification": 3.1})

    def test_validation_and_uq_capped_two_above_verification(self) -> None:
        """Section 4.2.5: code 1 and solution 0 allow at most level 2.

        References:
            [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 4.2.5].
            Checks [EQ-VAL-CRED-003].

        """
        scale = pcmm_scale()
        base = {"Code Verification": 1, "Solution Verification": 0}
        ok = _assess(scale, {**base, "Model Validation": 2})
        self.assertEqual(ok.level("Model Validation"), 2)
        for element in (
            "Model Validation",
            "Uncertainty Quantification and Sensitivity Analysis",
        ):
            with self.assertRaises(ValueError) as caught:
                _assess(scale, {**base, element: 3})
            self.assertIn("exceeds the limit", str(caught.exception))
        raised = {"Code Verification": 3, "Solution Verification": 2}
        _assess(scale, {**raised, "Model Validation": 3})

    def test_cap_is_not_enforced_when_a_driving_element_is_unassessed(self) -> None:
        """An unassessed verification element leaves the cap undecidable.

        References:
            [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 4.2.5].
            Checks [EQ-VAL-CRED-003].

        """
        levels = {"Solution Verification": None, "Model Validation": 3}
        assessment = _assess(pcmm_scale(), levels)
        self.assertEqual(assessment.level("Model Validation"), 3)


class Cas2008Tests(unittest.TestCase):
    """The eight-factor 2008 scale and its minimum roll-up.

    References:
        [REF-BABULA-ETAL-2009, Section IV]. Checks [EQ-VAL-CRED-001],
        [EQ-VAL-CRED-004].

    """

    def test_eight_factors_in_three_categories(self) -> None:
        """Section IV lists eight factors in three named categories.

        References:
            [REF-BABULA-ETAL-2009, Section IV].

        """
        scale = nasa_cas_2008_scale()
        self.assertEqual(list(scale.factor_names), CAS_2008)
        self.assertEqual(
            scale.categories,
            ("M&S Development", "M&S Operations", "Supporting Evidence"),
        )
        sizes = [len(scale.factors_in(c)) for c in scale.categories]
        self.assertEqual(sizes, [2, 3, 3])
        for factor in scale.factors:
            self.assertEqual(factor.max_level, 4)
        evidence_review = [f.name for f in scale.factors if f.subfactors]
        self.assertEqual(evidence_review, CAS_2008[:5])
        self.assertEqual(scale.source_key, "REF-BABULA-ETAL-2009")

    def test_overall_is_the_minimum_with_its_weakest_factor(self) -> None:
        """Hand example: levels 3,2,4,1,2,3,3,4 give overall 1.

        References:
            [REF-BABULA-ETAL-2009, Roll-up Processes]. Checks [EQ-VAL-CRED-001].

        """
        scale = nasa_cas_2008_scale()
        levels = dict(zip(CAS_2008, (3, 2, 4, 1, 2, 3, 3, 4), strict=True))
        assessment = _assess(scale, levels)
        overall = assessment.overall()
        self.assertEqual(overall.rule, "minimum")
        self.assertEqual(overall.minimum, 1)
        self.assertEqual(overall.weakest, ("Results Uncertainty",))
        self.assertIsNone(overall.mean)
        self.assertEqual(
            assessment.category_minimum(),
            {"M&S Development": 2, "M&S Operations": 1, "Supporting Evidence": 3},
        )

    def test_partial_assessment_needs_allow_partial(self) -> None:
        """A not-assessed factor blocks the roll-up unless partial is allowed.

        References:
            [REF-BABULA-ETAL-2009, Roll-up Processes]. Checks [EQ-VAL-CRED-001].

        """
        levels = dict(zip(CAS_2008, (3, 2, 4, None, 2, 3, 3, 4), strict=True))
        assessment = _assess(nasa_cas_2008_scale(), levels)
        with self.assertRaises(ValueError):
            assessment.overall()
        with self.assertRaises(ValueError):
            assessment.category_minimum()
        self.assertEqual(assessment.overall(allow_partial=True).minimum, 2)
        none_assessed = {name: None for name in CAS_2008}
        empty = _assess(nasa_cas_2008_scale(), none_assessed)
        with self.assertRaises(ValueError):
            empty.overall(allow_partial=True)

    def test_weighted_evidence_review_level(self) -> None:
        """The paper's extreme case, evidence 0 and review 4, gives 1.2.

        References:
            [REF-BABULA-ETAL-2009, Roll-up Processes]. Checks [EQ-VAL-CRED-004].

        """
        self.assertAlmostEqual(weighted_evidence_review_level(0, 4, 0.3), 1.2)
        self.assertAlmostEqual(weighted_evidence_review_level(4, 0, 0.3), 2.8)
        self.assertEqual(weighted_evidence_review_level(3, 4, 0.0), 3.0)
        with self.assertRaises(ValueError):
            weighted_evidence_review_level(2, 3, 0.31)
        with self.assertRaises(ValueError):
            weighted_evidence_review_level(2, 3, -0.1)
        with self.assertRaises(ValueError):
            weighted_evidence_review_level(-1, 3, 0.2)

    def test_weighted_factor_score_is_accepted_as_a_level(self) -> None:
        """The 2008 scale accepts any real level inside its range.

        References:
            [REF-BABULA-ETAL-2009, Roll-up Processes]. Checks [EQ-VAL-CRED-004].

        """
        value = weighted_evidence_review_level(2, 3, 0.25)
        assessment = _assess(nasa_cas_2008_scale(), {"Verification": value})
        self.assertAlmostEqual(assessment.level("Verification") or 0.0, 2.25)


class CriticalityTests(unittest.TestCase):
    """The criticality matrix of Appendix D, Figure 1.

    References:
        [REF-NASA-STD-7009B-2024, Appendix D]. Checks [EQ-VAL-CRED-006].

    """

    def test_matrix_rows_match_the_rendered_figure(self) -> None:
        """Rows influence 5..1, columns consequence I..V, as in Figure 1.

        References:
            [REF-NASA-STD-7009B-2024, Appendix D, Figure 1].
            Checks [EQ-VAL-CRED-006].

        """
        rows = {5: "GYRRR", 4: "GYYRR", 3: "GYYYR", 2: "GGGYY", 1: "GGGGY"}
        matrix = criticality_matrix()
        self.assertEqual(len(matrix), 25)
        for influence, text in rows.items():
            for consequence, rating in enumerate(text, start=1):
                self.assertEqual(matrix[(consequence, influence)], rating)
                found = criticality_assessment(consequence, influence)
                self.assertEqual(found.rating, rating)

    def test_matrix_is_monotone_and_has_the_stated_counts(self) -> None:
        """Ratings never decrease with either axis; 6 red, 9 yellow, 10 green.

        References:
            [REF-NASA-STD-7009B-2024, Appendix D, Figure 1].
            Checks [EQ-VAL-CRED-006].

        """
        order = {"G": 0, "Y": 1, "R": 2}
        matrix = criticality_matrix()
        for c in range(1, 6):
            for i in range(1, 6):
                here = order[matrix[(c, i)]]
                if c < 5:
                    self.assertLessEqual(here, order[matrix[(c + 1, i)]])
                if i < 5:
                    self.assertLessEqual(here, order[matrix[(c, i + 1)]])
        counts = {k: list(matrix.values()).count(k) for k in "RYG"}
        self.assertEqual(counts, {"R": 6, "Y": 9, "G": 10})

    def test_meaning_labels_and_validation(self) -> None:
        """Red is a candidate for full adherence; invalid levels are rejected.

        References:
            [REF-NASA-STD-7009B-2024, Appendix D.1]. Checks [EQ-VAL-CRED-006].

        """
        red = criticality_assessment(5, 5)
        self.assertEqual(red.rating, "R")
        self.assertIn("completely adhering", red.meaning)
        self.assertEqual(red.consequence_label, "V: Catastrophic")
        self.assertEqual(red.influence_label, "5: Controlling")
        self.assertIn("discretion", criticality_assessment(3, 3).meaning)
        self.assertIn("no critical", criticality_assessment(1, 1).meaning)
        for bad in (0, 6):
            with self.assertRaises(ValueError):
                criticality_assessment(bad, 3)
            with self.assertRaises(ValueError):
                criticality_assessment(3, bad)
        with self.assertRaises(TypeError):
            criticality_assessment(2.0, 3)  # type: ignore[arg-type]


class GapAnalysisTests(unittest.TestCase):
    """Target profiles, gap lists, and the improvement order.

    References:
        [REF-NASA-STD-7009B-2024, Section 4.3.8.4]. Checks [EQ-VAL-CRED-005].

    """

    def setUp(self) -> None:
        """Build a 7009B assessment with known achieved levels.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        self.scale = nasa_std_7009b_scale()
        self.assessment = _assess(
            self.scale,
            {
                "M&S Data Pedigree": 3,
                "M&S Verification": 2,
                "M&S Validation": 1,
                "M&S Uncertainty Characterization": 2,
                "M&S Results Robustness": None,
            },
        )
        self.profile = TargetProfile(
            "decision under review",
            {
                "M&S Data Pedigree": 3,
                "M&S Verification": 3,
                "M&S Validation": 3,
                "M&S Uncertainty Characterization": 3,
                "M&S Results Robustness": 2,
                "M&S Input Pedigree": 0,
            },
            criticality="R",
            basis="test thresholds",
        )

    def test_gap_list_values(self) -> None:
        """Met factors and zero requirements give no gap; others show shortfall.

        References:
            [REF-NASA-STD-7009B-2024, Section 4.3.8.4]. Checks [EQ-VAL-CRED-005].

        """
        gaps = level_gaps(self.assessment, self.profile)
        by_name = {g.factor: g for g in gaps}
        self.assertEqual(
            set(by_name),
            {
                "M&S Verification",
                "M&S Validation",
                "M&S Uncertainty Characterization",
                "M&S Results Robustness",
            },
        )
        self.assertEqual(by_name["M&S Verification"].shortfall, 1)
        self.assertEqual(by_name["M&S Validation"].shortfall, 2)
        self.assertIsNone(by_name["M&S Results Robustness"].shortfall)
        self.assertIsNone(by_name["M&S Results Robustness"].achieved)
        verification = self.scale.factor("M&S Verification")
        self.assertEqual(
            by_name["M&S Verification"].next_level, verification.levels[3].descriptor
        )
        self.assertFalse(is_sufficient(self.assessment, self.profile))
        order = [g.factor for g in gaps]
        self.assertEqual(order, [n for n in self.scale.factor_names if n in by_name])

    def test_improvement_order_unassessed_first_then_largest_gap(self) -> None:
        """Unassessed factors first, then decreasing shortfall, ties in order.

        References:
            [REF-NASA-STD-7009B-2024, Section 4.3.8.4]. Checks [EQ-VAL-CRED-005].

        """
        steps = improvement_plan(self.assessment, self.profile)
        self.assertEqual(
            [s.factor for s in steps],
            [
                "M&S Results Robustness",
                "M&S Validation",
                "M&S Verification",
                "M&S Uncertainty Characterization",
            ],
        )
        self.assertEqual([s.rank for s in steps], [1, 2, 3, 4])
        self.assertIn("assess this factor first", steps[0].action)
        self.assertEqual(steps[1].from_level, 1)
        self.assertEqual(steps[1].to_level, 3)

    def test_sufficient_profile_gives_no_gap(self) -> None:
        """Meeting or exceeding every requirement yields an empty list.

        References:
            [REF-NASA-STD-7009B-2024, Section 4.3.8.4]. Checks [EQ-VAL-CRED-005].

        """
        top = _assess(self.scale, {n: 4 for n in self.scale.factor_names})
        profile = uniform_profile(self.scale, 3, name="all three")
        self.assertEqual(level_gaps(top, profile), ())
        self.assertTrue(is_sufficient(top, profile))
        self.assertEqual(improvement_plan(top, profile), ())
        self.assertIn("No gap", render_gaps(()))
        self.assertIn("Nothing to improve", render_improvement_plan(()))

    def test_uniform_profile_clips_to_each_factor_range(self) -> None:
        """A uniform requirement above a factor's maximum is clipped to it.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        profile = uniform_profile(pcmm_scale(), 5, name="max")
        self.assertEqual(set(profile.required.values()), {3.0})
        with self.assertRaises(ValueError):
            uniform_profile(pcmm_scale(), -1, name="bad")

    def test_profile_with_unknown_factor_or_level_is_rejected(self) -> None:
        """A profile that does not fit the scale raises.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        with self.assertRaises(ValueError):
            level_gaps(self.assessment, TargetProfile("x", {"No such factor": 1}))
        with self.assertRaises(ValueError):
            level_gaps(self.assessment, TargetProfile("x", {"M&S Verification": 5}))
        with self.assertRaises(ValueError):
            TargetProfile("x", {})
        with self.assertRaises(ValueError):
            TargetProfile("x", {"M&S Verification": -1})
        with self.assertRaises(TypeError):
            level_gaps(self.assessment, {"M&S Verification": 3})  # type: ignore[arg-type]

    def test_pcmm_cap_prerequisites_are_listed(self) -> None:
        """Raising validation to 3 first needs solution verification above 0.

        References:
            [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 4.2.5].
            Checks [EQ-VAL-CRED-003], [EQ-VAL-CRED-005].

        """
        levels = {
            "Code Verification": 1,
            "Solution Verification": 0,
            "Model Validation": 2,
        }
        assessment = _assess(pcmm_scale(), levels)
        profile = TargetProfile("qualification", {"Model Validation": 3})
        (step,) = improvement_plan(assessment, profile)
        self.assertEqual(step.prerequisites, ("Solution Verification",))
        self.assertEqual(step.shortfall, 1)


class AssessmentValidationTests(unittest.TestCase):
    """Rejection of unknown factors, levels, duplicates, and bad definitions.

    References:
        [REF-NASA-STD-7009B-2024, Section 4.3.6.1].

    """

    def test_unknown_duplicate_and_unexplained_not_assessed(self) -> None:
        """Unknown or repeated factors and silent skips are rejected.

        References:
            [REF-NASA-STD-7009B-2024, Section 4.3.6.1].

        """
        scale = pcmm_scale()
        good = list(_scores(scale, {}))
        with self.assertRaises(ValueError):
            CredibilityAssessment(scale, (*good, FactorScore("Nonsense", 1)))
        with self.assertRaises(ValueError):
            CredibilityAssessment(scale, (*good, good[0]))
        with self.assertRaises(ValueError):
            FactorScore("Code Verification", None)
        with self.assertRaises(ValueError):
            FactorScore("Code Verification", -1)
        with self.assertRaises(TypeError):
            CredibilityAssessment("scale", ())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            CredibilityAssessment(scale, ("score",))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            scale.factor("Nonsense")
        with self.assertRaises(ValueError):
            scale.factors_in("Nonsense")

    def test_subfactor_scores_are_checked(self) -> None:
        """Sub-factor scores need a declared sub-factor and a valid level.

        References:
            [REF-BABULA-ETAL-2009, Roll-up Processes].

        """
        scale = nasa_cas_2008_scale()
        scores = list(_scores(scale, {}))
        scores[0] = FactorScore(
            "Verification",
            2,
            rationale="ok",
            subscores={"Evidence": 2, "Technical Review": 3},
        )
        CredibilityAssessment(scale, tuple(scores))
        for sub in ({"Weight": 1}, {"Evidence": 9}):
            scores[0] = FactorScore("Verification", 2, rationale="ok", subscores=sub)
            with self.assertRaises(ValueError):
                CredibilityAssessment(scale, tuple(scores))

    def test_scale_definition_is_validated(self) -> None:
        """Duplicate names, bad roll-up, bad caps, and bad levels are rejected.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        levels = (CredibilityLevel(0, "none"), CredibilityLevel(1, "some"))
        factor = CredibilityFactor("A", "cat", levels)

        def build(**changes: object) -> CredibilityScale:
            """Build a small scale with some fields replaced.

            References:
                [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

            """
            fields: dict[str, object] = {
                "key": "k",
                "title": "t",
                "source_key": "REF-X",
                "citation": "c",
                "sections": ("s",),
                "factors": (factor,),
            }
            fields.update(changes)
            return CredibilityScale(**fields)  # type: ignore[arg-type]

        self.assertEqual(build().rollup, "none")
        with self.assertRaises(ValueError):
            build(factors=(factor, factor))
        with self.assertRaises(ValueError):
            build(rollup="mean")
        with self.assertRaises(ValueError):
            build(caps=(LevelCap(("A",), ("B",), 1),))
        with self.assertRaises(ValueError):
            build(factors=())
        with self.assertRaises(ValueError):
            build(resolution=-0.1)
        with self.assertRaises(ValueError):
            CredibilityFactor("A", "cat", levels[:1])
        with self.assertRaises(ValueError):
            CredibilityFactor("A", "cat", (levels[1], levels[0]))
        with self.assertRaises(ValueError):
            CredibilityFactor("A", "cat", levels, record_evidence=("gossip",))
        with self.assertRaises(ValueError):
            CredibilityLevel(-1, "x")
        with self.assertRaises(TypeError):
            CredibilityLevel(1.0, "x")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            LevelCap((), ("A",), 1)
        with self.assertRaises(ValueError):
            LevelCap(("A",), ("A",), -1)

    def test_custom_scale_works_end_to_end(self) -> None:
        """A user-supplied scale with a minimum roll-up behaves like the shipped ones.

        References:
            [REF-PROBSTAT-VALIDATION]. Checks [EQ-VAL-CRED-001].

        """
        low_high = (CredibilityLevel(0, "none"), CredibilityLevel(1, "some"))
        scale = CredibilityScale(
            "custom",
            "Custom scale",
            "REF-X",
            "own scale",
            ("n/a",),
            (
                CredibilityFactor("A", "one", low_high),
                CredibilityFactor("B", "one", low_high),
                CredibilityFactor("C", "two", low_high),
            ),
            rollup="minimum",
        )
        assessment = _assess(scale, {"A": 1, "B": 0, "C": 1})
        self.assertEqual(assessment.overall().weakest, ("B",))
        self.assertEqual(assessment.category_minimum(), {"one": 0, "two": 1})


class BridgeTests(unittest.TestCase):
    """Record fields are listed per factor, never turned into levels.

    References:
        [REF-ROY-OBERKAMPF-2011]. No mathematical equation is evaluated.

    """

    def _record(self) -> CredibilityRecord:
        """Build a record with verification and prediction evidence only.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        verification = VerificationEvidence(
            "y",
            NumericalErrorBudget(discretization=0.003, iterative=0.001),
            "three-grid refinement study",
            observed_order=1.98,
        )
        prediction = PredictionStatement(
            "y", {"load": 8.0}, 1.0, 2.0, 0.95, "input uncertainty only"
        )
        return CredibilityRecord((verification,), (), (prediction,))

    def test_suggestions_name_matching_evidence_and_empty_kinds(self) -> None:
        """Verification and prediction items appear; validation is empty.

        References:
            [REF-ROY-OBERKAMPF-2011]. No mathematical equation is evaluated.

        """
        scale = nasa_std_7009b_scale()
        suggestions = suggest_record_evidence(self._record(), scale)
        found = {(s.factor, s.kind): s for s in suggestions}
        verification = found[("M&S Verification", "verification")]
        self.assertEqual(len(verification.items), 1)
        self.assertIn("observed order 1.98", verification.items[0])
        self.assertIn("0.004", verification.items[0])
        self.assertEqual(found[("M&S Validation", "validation")].items, ())
        uncertainty = found[("M&S Uncertainty Characterization", "prediction")]
        self.assertIn("[1, 2]", uncertainty.items[0])
        domain = found[("M&S Use Assessment", "domain")]
        self.assertIn("domain not checked", domain.items[0])
        self.assertTrue(all("does not set a level" in s.note for s in suggestions))
        factors = {s.factor for s in suggestions}
        self.assertNotIn("M&S Input Pedigree", factors)
        self.assertNotIn("M&S Development Technical Review", factors)

    def test_pcmm_and_type_errors(self) -> None:
        """PCMM verification elements both receive the budget; types are checked.

        References:
            [REF-ROY-OBERKAMPF-2011]. No mathematical equation is evaluated.

        """
        suggestions = suggest_record_evidence(self._record(), pcmm_scale())
        named = {s.factor for s in suggestions if s.kind == "verification"}
        self.assertEqual(named, {"Code Verification", "Solution Verification"})
        with self.assertRaises(TypeError):
            suggest_record_evidence("record", pcmm_scale())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            suggest_record_evidence(self._record(), "scale")  # type: ignore[arg-type]

    def test_suggestions_do_not_alter_the_record(self) -> None:
        """Calling the bridge leaves the record and its gaps unchanged.

        References:
            [REF-ROY-OBERKAMPF-2011]. No mathematical equation is evaluated.

        """
        record = self._record()
        before = (record.gaps, record.summary())
        suggest_record_evidence(record, nasa_std_7009b_scale())
        self.assertEqual((record.gaps, record.summary()), before)


class RenderingAndArchiveTests(unittest.TestCase):
    """Plain-text and Markdown rendering and the ``to_dict`` round trip.

    References:
        [REF-NASA-STD-7009B-2024, Appendix E.5].

    """

    def _assessment(self) -> CredibilityAssessment:
        """Build a 7009B assessment with evidence, text fields, and a gap.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        scale = nasa_std_7009b_scale()
        scores = list(_scores(scale, {"M&S Verification": 2}))
        scores[1] = FactorScore(
            "M&S Verification",
            2,
            ("V&V report 12", "regression log"),
            "documented practices exist",
            assessor="reviewer A",
            date="2026-11-24",
        )
        return CredibilityAssessment(
            scale,
            tuple(scores),
            subject="trajectory model v3",
            intended_use="entry corridor check",
        )

    def test_text_and_markdown_tables(self) -> None:
        """Both renderings list every factor, level, descriptor, and profile.

        References:
            [REF-NASA-STD-7009B-2024, Appendix E.5].

        """
        assessment = self._assessment()
        profile = TargetProfile("p", {"M&S Verification": 3})
        text = render_assessment(assessment, profile)
        self.assertIn("trajectory model v3", text)
        self.assertIn("entry corridor check", text)
        for name in assessment.scale.factor_names:
            self.assertIn(name, text)
        self.assertIn("Required", text.splitlines()[4])
        markdown = render_assessment(assessment, profile, markdown=True)
        self.assertTrue(markdown.startswith("## "))
        self.assertIn("| Category | Factor | Level | Required |", markdown)
        rows = [ln for ln in markdown.splitlines() if ln.startswith("| ")]
        self.assertEqual(len(rows), 1 + 1 + 11)
        gaps = level_gaps(assessment, profile)
        gap_text = render_gaps(gaps)
        self.assertIn("M&S Verification", gap_text)
        self.assertIn("Shortfall", gap_text)
        gap_md = render_gaps(gaps, markdown=True)
        self.assertIn("| M&S Verification | 2 | 3 | 1 |", gap_md)
        plan_md = render_improvement_plan(
            improvement_plan(assessment, profile), markdown=True
        )
        self.assertIn("| 1 | M&S Verification | 2 | 3 |", plan_md)

    def test_pipes_are_escaped_and_partial_levels_marked(self) -> None:
        """A pipe in text does not break a Markdown row; 1.5 reads as partial.

        References:
            [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 4.1].

        """
        levels = (CredibilityLevel(0, "a | b"), CredibilityLevel(1, "c"))
        scale = CredibilityScale(
            "k",
            "t",
            "REF-X",
            "c",
            ("s",),
            (CredibilityFactor("F | G", "cat", levels),),
            resolution=0.5,
        )
        score = FactorScore("F | G", 0.5, rationale="half")
        markdown = render_assessment(
            CredibilityAssessment(scale, (score,)), markdown=True
        )
        self.assertIn("F \\| G", markdown)
        self.assertIn("a \\| b (partly toward level 1)", markdown)
        pcmm = _assess(pcmm_scale(), {"Code Verification": 1.5})
        self.assertIn("(partly toward level 2)", render_assessment(pcmm))

    def test_render_reports_the_defined_rollup(self) -> None:
        """Minimum and triple scales print their roll-up line.

        References:
            [REF-BABULA-ETAL-2009, Roll-up Processes].
            [REF-OBERKAMPF-PILCH-TRUCANO-2007, Section 5.2].

        """
        cas = _assess(nasa_cas_2008_scale(), {n: 2 for n in CAS_2008})
        self.assertIn("Overall (minimum): 2", render_assessment(cas))
        pcmm = _assess(pcmm_scale(), {"Code Verification": 1})
        line = render_assessment(pcmm).splitlines()[-1]
        self.assertIn("(minimum, mean, maximum): (0, 0.167, 1)", line)

    def test_archive_round_trip_through_json(self) -> None:
        """Scale, assessment, and profile survive ``to_dict`` and JSON.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        for scale in (nasa_std_7009b_scale(), pcmm_scale(), nasa_cas_2008_scale()):
            rebuilt = CredibilityScale.from_dict(
                json.loads(json.dumps(scale.to_dict()))
            )
            self.assertEqual(rebuilt, scale)
        assessment = self._assessment()
        data = json.loads(json.dumps(assessment.to_dict()))
        restored = CredibilityAssessment.from_dict(data)
        self.assertEqual(restored, assessment)
        self.assertEqual(restored.to_dict(), assessment.to_dict())
        self.assertEqual(restored.score("M&S Verification").assessor, "reviewer A")
        profile = TargetProfile("p", {"M&S Verification": 3}, "R", "basis")
        self.assertEqual(TargetProfile.from_dict(profile.to_dict()), profile)
        pcmm = _assess(pcmm_scale(), {"Code Verification": 1.5})
        again = CredibilityAssessment.from_dict(json.loads(json.dumps(pcmm.to_dict())))
        self.assertEqual(again.level("Code Verification"), 1.5)

    def test_archive_rejects_missing_keys(self) -> None:
        """Incomplete dictionaries raise ``ValueError``, not ``KeyError``.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        data = self._assessment().to_dict()
        with self.assertRaises(ValueError):
            CredibilityAssessment.from_dict({"scale": data["scale"]})
        with self.assertRaises(ValueError):
            CredibilityScale.from_dict({"key": "k"})
        with self.assertRaises(ValueError):
            TargetProfile.from_dict({"name": "p"})
        broken = dict(data["scale"])
        del broken["factors"]
        with self.assertRaises(ValueError):
            CredibilityScale.from_dict(broken)

    def test_every_source_key_is_in_the_bibliography(self) -> None:
        """Each shipped scale cites a key that exists in ``docs/references.md``.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        text = (ROOT / "docs" / "references.md").read_text(encoding="utf-8")
        keys = set(re.findall(r"^## (REF-[A-Z0-9-]+)\s*$", text, re.MULTILINE))
        for scale in (nasa_std_7009b_scale(), pcmm_scale(), nasa_cas_2008_scale()):
            self.assertIn(scale.source_key, keys)


if __name__ == "__main__":
    unittest.main()
