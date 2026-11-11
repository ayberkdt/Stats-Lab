"""Small historical data sets with exact provenance for real-data case studies.

Each record is frozen and carries its values, units, the original
publication with a page or table locator, a secondary source that was checked
against the values, and a licence statement. The values were transcribed from
the cited pages and are checked by ``tests/test_real_data_case_studies.py``
against the totals and summaries printed in those sources (ADR-0080).
"""

from __future__ import annotations

from dataclasses import dataclass

from probstat.foundations import coerce_finite_sample


@dataclass(frozen=True, slots=True)
class DataColumn:
    """One named variable of a data set with its unit.

    Args:
        name: Identifier of the variable (lower case, underscores).
        unit: Unit or counting rule of every value.
        values: Finite values in the order of the source.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If the name or unit is empty or a value is not finite.

    References:
        [REF-NIST-EDA, Section 1.3]. No equation is evaluated.

    """

    name: str
    unit: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the name, unit, and values and store the values as floats.

        Raises:
            TypeError: If the name or unit is not text or a value is not real.
            ValueError: If the name or unit is empty or a value is not finite.

        References:
            [REF-NIST-EDA, Section 1.3]. No equation is evaluated.

        """
        for label, text in (("name", self.name), ("unit", self.unit)):
            if not isinstance(text, str):
                raise TypeError(f"{label} must be a string")
            if not text.strip():
                raise ValueError(f"{label} must not be empty")
        object.__setattr__(
            self, "values", coerce_finite_sample(self.values, name=self.name)
        )


@dataclass(frozen=True, slots=True)
class DataSet:
    """A frozen historical data set with provenance and licence.

    Args:
        key: Short identifier used by :func:`dataset_by_key`.
        title: Descriptive title.
        description: What was measured or counted, and how the columns relate.
        columns: Variables of equal length, in the order of the source table.
        primary_source: Original publication with a page or table locator.
        secondary_source: Independent source against which the values were
            checked, with what it confirms.
        licence: Why the values may be redistributed.
        references: Bibliography keys of the primary and secondary sources.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If there are no columns, names repeat, lengths differ, or a
            text field is empty.

    References:
        [REF-NIST-EDA, Section 1.3]. No equation is evaluated.

    """

    key: str
    title: str
    description: str
    columns: tuple[DataColumn, ...]
    primary_source: str
    secondary_source: str
    licence: str
    references: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the record structure.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If the record is empty or inconsistent.

        References:
            [REF-NIST-EDA, Section 1.3]. No equation is evaluated.

        """
        texts = {
            "key": self.key,
            "title": self.title,
            "description": self.description,
            "primary_source": self.primary_source,
            "secondary_source": self.secondary_source,
            "licence": self.licence,
        }
        for label, text in texts.items():
            if not isinstance(text, str):
                raise TypeError(f"{label} must be a string")
            if not text.strip():
                raise ValueError(f"{label} must not be empty")
        columns = tuple(self.columns)
        if not columns:
            raise ValueError("a data set needs at least one column")
        if not all(isinstance(column, DataColumn) for column in columns):
            raise TypeError("columns must be DataColumn records")
        names = [column.name for column in columns]
        if len(set(names)) != len(names):
            raise ValueError("column names must be distinct")
        if len({len(column.values) for column in columns}) != 1:
            raise ValueError("all columns must have the same length")
        references = tuple(self.references)
        if not references or not all(
            isinstance(key, str) and key.startswith("REF-") for key in references
        ):
            raise ValueError("references must be non-empty REF-* keys")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "references", references)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return the column names in source order.

        References:
            [REF-NIST-EDA, Section 1.3]. No equation is evaluated.

        """
        return tuple(column.name for column in self.columns)

    @property
    def rows(self) -> int:
        """Return the number of rows of the source table.

        References:
            [REF-NIST-EDA, Section 1.3]. No equation is evaluated.

        """
        return len(self.columns[0].values)

    def column(self, name: str) -> DataColumn:
        """Return the column with the given name.

        Args:
            name: Column name.

        Returns:
            The matching column.

        Raises:
            KeyError: If no column has that name.

        References:
            [REF-NIST-EDA, Section 1.3]. No equation is evaluated.

        """
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"{self.key} has no column {name!r}")

    def values(self, name: str | None = None) -> tuple[float, ...]:
        """Return the values of one column (the first column by default).

        Args:
            name: Column name, or ``None`` for the first column.

        Returns:
            Column values in source order.

        Raises:
            KeyError: If no column has that name.

        References:
            [REF-NIST-EDA, Section 1.3]. No equation is evaluated.

        """
        if name is None:
            return self.columns[0].values
        return self.column(name).values

    def expand_frequencies(self, value: str, frequency: str) -> tuple[int, ...]:
        """Expand a frequency table into one integer observation per unit.

        Args:
            value: Column holding the non-negative integer outcomes ``k``.
            frequency: Column holding how many units showed each outcome.

        Returns:
            Ascending tuple in which outcome ``k`` appears ``f_k`` times; its
            length is ``sum_k f_k``.

        Raises:
            KeyError: If a column is missing.
            ValueError: If an outcome or a frequency is not a non-negative
                integer.

        References:
            [REF-NIST-EDA, Section 1.3]. No equation is evaluated.

        """
        outcomes = self.column(value).values
        counts = self.column(frequency).values
        sample: list[int] = []
        for outcome, count in sorted(zip(outcomes, counts, strict=True)):
            if outcome < 0 or count < 0 or not (
                float(outcome).is_integer() and float(count).is_integer()
            ):
                raise ValueError(
                    "outcomes and frequencies must be non-negative integers"
                )
            sample.extend([int(outcome)] * int(count))
        return tuple(sample)


def _column(name: str, unit: str, values: tuple[float, ...]) -> DataColumn:
    """Build a column (a short alias used by the records below).

    Args:
        name: Column name.
        unit: Unit text.
        values: Column values.

    Returns:
        Validated column.

    References:
        [REF-NIST-EDA, Section 1.3]. No equation is evaluated.

    """
    return DataColumn(name=name, unit=unit, values=values)


_PRE_1930_LICENCE = (
    "Public domain. The values were published in {year}, before 1930, so no "
    "United States copyright subsists in the publication; the values are "
    "observed counts or measurements, which are facts and are not protected "
    "by copyright in any case."
)

BORTKIEWICZ_HORSE_KICKS = DataSet(
    key="bortkiewicz-horse-kicks",
    title="Deaths by horse kick in ten Prussian army corps over twenty years",
    description=(
        "Number of soldiers killed by the kick of a horse per corps and "
        "calendar year, pooled over the ten army corps that remain after "
        "Bortkiewicz set aside the Guard and the I, VI, and XI corps for their "
        "atypical composition: 10 corps x 20 years = 200 corps-years with 122 "
        "deaths. Column 'deaths' is the annual count k; column 'corps_years' "
        "is the number of corps-years with exactly k deaths."
    ),
    columns=(
        _column("deaths", "deaths per corps-year", (0, 1, 2, 3, 4)),
        _column("corps_years", "corps-years", (109, 65, 22, 3, 1)),
    ),
    primary_source=(
        "L. von Bortkewitsch, Das Gesetz der kleinen Zahlen (Leipzig: B. G. "
        "Teubner, 1898), Chapter 2, Section 12 ('4. Beispiel'), case b), "
        "table on page 25 (observed column; the expected column of the same "
        "table lists 108.7, 66.3, 20.2, 4.1, 0.6, 0.1)."
    ),
    secondary_source=(
        "Checked against the Internet Archive scan of the 1898 Teubner edition "
        "(item dasgesetzderklei00bortrich, page 25) and the transcription of "
        "Section 12 in the University of York history-of-statistics pages, "
        "which gives the same observed column 109, 65, 22, 3, 1."
    ),
    licence=_PRE_1930_LICENCE.format(year=1898),
    references=("REF-BORTKIEWICZ-1898", "REF-YORK-BORTKIEWICZ-TRANSCRIPTION"),
)

RUTHERFORD_GEIGER_SCINTILLATIONS = DataSet(
    key="rutherford-geiger-scintillations",
    title="Alpha-particle scintillations per eighth of a minute",
    description=(
        "Number of alpha particles counted by the scintillation method in "
        "successive intervals of 1/8 minute (7.5 s), "
        "combined over the four parts of the recording tape: 2608 intervals "
        "with 10097 particles. Column 'particles' is the count k; column "
        "'intervals' is the number of intervals with exactly k particles."
    ),
    columns=(
        _column(
            "particles",
            "alpha particles per 7.5 s interval",
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
        ),
        _column(
            "intervals",
            "intervals",
            (57, 203, 383, 525, 532, 408, 273, 139, 45, 27, 10, 4, 0, 1, 1),
        ),
    ),
    primary_source=(
        "E. Rutherford and H. Geiger, with a note by H. Bateman, 'The "
        "probability variations in the distribution of alpha particles', "
        "Philosophical Magazine, Series 6, 20(118), 698-707 (1910), table on "
        "page 701, row 'Sum' (totals 10097 particles, 2608 intervals, average "
        "number 3.87)."
    ),
    secondary_source=(
        "Checked against the scan of page 701 distributed with J. A. Hanley's "
        "teaching page on the experiment (McGill University), whose text also "
        "quotes the totals 2608 intervals and 10,097 particles and the "
        "525 intervals with three particles."
    ),
    licence=_PRE_1930_LICENCE.format(year=1910),
    references=("REF-RUTHERFORD-GEIGER-1910", "REF-HANLEY-RUTHERFORD-GEIGER"),
)

NEWCOMB_PASSAGE_TIMES = DataSet(
    key="newcomb-passage-times",
    title="Newcomb's third series of light passage times, 1882",
    description=(
        "Sixty-six measurements of the time light took to travel to a distant "
        "mirror and back (Newcomb's third series), recorded as deviations: "
        "the passage time in nanoseconds is "
        "24800 + value (equivalently 24.8 + value/1000 microseconds). Two "
        "values (-44 and -2) are well-known outliers; the value 33.02 is the "
        "deviation corresponding to the currently accepted speed of light."
    ),
    columns=(
        _column(
            "deviation",
            "ns (passage time minus 24800 ns)",
            (
                28, 26, 33, 24, 34, -44, 27, 16, 40, -2, 29, 22, 24, 21, 25,
                30, 23, 29, 31, 19, 24, 20, 36, 32, 36, 28, 25, 21, 28, 29,
                37, 25, 28, 26, 30, 32, 36, 26, 30, 22, 36, 23, 27, 27, 28,
                27, 31, 27, 26, 33, 26, 32, 32, 24, 39, 28, 24, 25, 32, 25,
                29, 27, 28, 29, 16, 23,
            ),
        ),
    ),
    primary_source=(
        "S. Newcomb, 'Measures of the velocity of light made under the "
        "direction of the Secretary of the Navy during the years 1880-1882', "
        "Astronomical Papers prepared for the use of the American Ephemeris "
        "and Nautical Almanac, Vol. II, Part III; reproduced in the order "
        "given by S. M. Stigler, 'Do robust estimators work with real data?', "
        "Annals of Statistics 5(6), 1055-1098 (1977), Table 5."
    ),
    secondary_source=(
        "Checked against the documentation of the 'newcomb' data set of the "
        "R package MASS, which lists Stigler's Table 5 order value by value, "
        "gives the conversion to passage time, and states the true value "
        "33.02. Stigler's table itself was not read directly."
    ),
    licence=(
        "Public domain. The measurements were made and published by the "
        "United States Navy in the nineteenth century (a work of the United "
        "States Government, 17 U.S.C. 105, and published before 1930); the "
        "values are measurements, which are facts and are not protected by "
        "copyright."
    ),
    references=(
        "REF-NEWCOMB-VELOCITY-OF-LIGHT",
        "REF-STIGLER-1977",
        "REF-R-MASS-NEWCOMB",
    ),
)

LIEBLEIN_ZELEN_BALL_BEARINGS = DataSet(
    key="lieblein-zelen-ball-bearings",
    title="Endurance of 23 deep-groove ball bearings",
    description=(
        "Fatigue endurance of the 23 bearings of one test group (lot size 25, "
        "two bearings struck out on the worksheet), in millions of "
        "revolutions, in ascending order. Worksheet conditions: 2000 r.p.m., "
        "jet-oil lubrication, nine balls of 1/4 in diameter."
    ),
    columns=(
        _column(
            "endurance",
            "millions of revolutions",
            (
                17.88, 28.92, 33.00, 41.52, 42.12, 45.60, 48.48, 51.84, 51.96,
                54.12, 55.56, 67.80, 68.64, 68.64, 68.88, 84.12, 93.12, 98.64,
                105.12, 105.84, 127.92, 128.04, 173.40,
            ),
        ),
    ),
    primary_source=(
        "J. Lieblein and M. Zelen, 'Statistical investigation of the fatigue "
        "life of deep-groove ball bearings', Journal of Research of the "
        "National Bureau of Standards 57(5), 273-316 (1956), Research Paper "
        "2719, 'Specimen worksheet' on page 286 (column 'Endurance Mill. "
        "Revs.', test no. 3183, 'Lot Size 25, Taken on 23')."
    ),
    secondary_source=(
        "The sample size and unit were checked against the documentation of "
        "the 'lzbearing' data set of the R package SMRD (23 rows, millions of "
        "cycles, citing Lawless 1982); the values were read from the NIST scan "
        "of the primary source."
    ),
    licence=(
        "Public domain. The paper was written by employees of the National "
        "Bureau of Standards as part of their official duties, so it is a "
        "work of the United States Government, which is not subject to "
        "copyright in the United States (17 U.S.C. 105); the values are "
        "measurements, which are facts."
    ),
    references=("REF-LIEBLEIN-ZELEN-1956", "REF-SMRD-LZBEARING"),
)

STUDENT_SLEEP = DataSet(
    key="student-sleep",
    title="Additional hours of sleep under two soporific drugs (ten patients)",
    description=(
        "Increase in hours of sleep, relative to control nights, of the same "
        "ten patients under each of two drugs (a paired design). The columns "
        "are named by position, as in Student's table; the paired difference "
        "is drug_2 - drug_1."
    ),
    columns=(
        _column("patient", "patient number", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
        _column(
            "drug_1",
            "hours of additional sleep",
            (0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0),
        ),
        _column(
            "drug_2",
            "hours of additional sleep",
            (1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4),
        ),
    ),
    primary_source=(
        "Student, 'The probable error of a mean', Biometrika 6(1), 1-25 "
        "(1908), table on page 20, quoting A. R. Cushny and A. R. Peebles, "
        "'The action of optical isomers. II. Hyoscines', Journal of "
        "Physiology 32(5-6), 501-510 (1905)."
    ),
    secondary_source=(
        "Checked against the 'sleep' data set of the R datasets package "
        "(columns 'extra', 'group', 'ID'), whose documentation cites Student "
        "(1908) page 20 and Cushny and Peebles (1905). The 1908 page itself "
        "was not read directly."
    ),
    licence=_PRE_1930_LICENCE.format(year="1905 and 1908"),
    references=(
        "REF-STUDENT-1908",
        "REF-CUSHNY-PEEBLES-1905",
        "REF-R-DATASETS-SLEEP",
    ),
)

DATASETS: tuple[DataSet, ...] = (
    BORTKIEWICZ_HORSE_KICKS,
    RUTHERFORD_GEIGER_SCINTILLATIONS,
    NEWCOMB_PASSAGE_TIMES,
    LIEBLEIN_ZELEN_BALL_BEARINGS,
    STUDENT_SLEEP,
)
"""Every bundled data set in a stable order."""


def dataset_by_key(key: str) -> DataSet:
    """Return a bundled data set by its key.

    Args:
        key: One of the ``DataSet.key`` values in :data:`DATASETS`.

    Returns:
        The matching data set.

    Raises:
        TypeError: If ``key`` is not text.
        KeyError: If no data set has that key.

    References:
        [REF-NIST-EDA, Section 1.3]. No equation is evaluated.

    """
    if not isinstance(key, str):
        raise TypeError("key must be a string")
    for dataset in DATASETS:
        if dataset.key == key:
            return dataset
    known = ", ".join(dataset.key for dataset in DATASETS)
    raise KeyError(f"unknown data set {key!r}; known keys: {known}")

