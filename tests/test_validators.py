# tests/test_validators.py
import numpy as np
import pandas as pd
import pytest

from src.data.validators.quality_checks import DataQualityChecker, QualityReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 10,
    with_geracao: bool = True,
    with_radiacao: bool = True,
    datetime_index: bool = True,
) -> pd.DataFrame:
    """Cria um DataFrame válido e limpo para usar nos testes."""
    index = (
        pd.date_range("2023-06-01 08:00", periods=n, freq="h")
        if datetime_index
        else range(n)
    )
    rng = np.random.default_rng(seed=0)
    data = {}
    if with_geracao:
        data["GERACAO"] = rng.uniform(0.1, 1.0, n)
    if with_radiacao:
        data["RADIACAO"] = rng.uniform(100, 800, n)
    data["TEMPERATURA"] = rng.uniform(20, 35, n)
    data["UMIDADE"] = rng.uniform(40, 90, n)
    return pd.DataFrame(data, index=index)


# ---------------------------------------------------------------------------
# QualityReport
# ---------------------------------------------------------------------------


class TestQualityReport:
    def test_passes_by_default(self):
        assert QualityReport().passed is True

    def test_issues_empty_by_default(self):
        assert QualityReport().issues == []

    def test_fail_marks_as_failed(self):
        report = QualityReport()
        report.fail("algo errado")
        assert report.passed is False

    def test_fail_appends_message(self):
        report = QualityReport()
        report.fail("problema 1")
        report.fail("problema 2")
        assert len(report.issues) == 2
        assert "problema 1" in report.issues

    def test_str_ok_when_passed(self):
        assert "OK" in str(QualityReport())

    def test_str_shows_issues_when_failed(self):
        report = QualityReport()
        report.fail("coluna ausente")
        text = str(report)
        assert "coluna ausente" in text
        assert "FALHOU" in text


# ---------------------------------------------------------------------------
# DataQualityChecker
# ---------------------------------------------------------------------------


class TestDataQualityCheckerAllPass:
    def test_clean_df_passes(self):
        report = DataQualityChecker().check(_make_df())
        assert report.passed

    def test_clean_df_has_no_issues(self):
        report = DataQualityChecker().check(_make_df())
        assert report.issues == []


class TestRequiredColumns:
    def test_fails_without_geracao(self):
        df = _make_df(with_geracao=False)
        report = DataQualityChecker().check(df)
        assert not report.passed
        assert any("GERACAO" in issue for issue in report.issues)

    def test_fails_without_radiacao(self):
        df = _make_df(with_radiacao=False)
        report = DataQualityChecker().check(df)
        assert not report.passed
        assert any("RADIACAO" in issue for issue in report.issues)

    def test_passes_with_both_required_columns(self):
        report = DataQualityChecker().check(_make_df())
        assert report.passed


class TestDuplicateTimestamps:
    def test_fails_with_duplicate_timestamps(self):
        df = _make_df()
        df = pd.concat([df, df.iloc[:2]])  # duplica as primeiras 2 linhas
        report = DataQualityChecker().check(df)
        assert not report.passed
        assert any("duplicado" in issue for issue in report.issues)

    def test_passes_with_unique_timestamps(self):
        report = DataQualityChecker().check(_make_df())
        assert report.passed

    def test_skips_check_for_non_datetime_index(self):
        df = _make_df(datetime_index=False)
        report = DataQualityChecker().check(df)
        # Sem DatetimeIndex, o check de duplicatas deve ser ignorado
        assert not any("duplicado" in issue for issue in report.issues)


class TestValueRanges:
    def test_fails_for_negative_radiacao(self):
        df = _make_df()
        df.loc[df.index[0], "RADIACAO"] = -1.0
        report = DataQualityChecker().check(df)
        assert not report.passed
        assert any("RADIACAO" in issue for issue in report.issues)

    def test_fails_for_radiacao_above_max(self):
        df = _make_df()
        df.loc[df.index[0], "RADIACAO"] = 99_999.0
        report = DataQualityChecker().check(df)
        assert not report.passed

    def test_fails_for_temperature_out_of_range(self):
        df = _make_df()
        df.loc[df.index[0], "TEMPERATURA"] = -100.0
        report = DataQualityChecker().check(df)
        assert not report.passed
        assert any("TEMPERATURA" in issue for issue in report.issues)

    def test_fails_for_humidity_above_100(self):
        df = _make_df()
        df.loc[df.index[0], "UMIDADE"] = 150.0
        report = DataQualityChecker().check(df)
        assert not report.passed
        assert any("UMIDADE" in issue for issue in report.issues)

    def test_fails_for_negative_geracao(self):
        df = _make_df()
        df.loc[df.index[0], "GERACAO"] = -0.5
        report = DataQualityChecker().check(df)
        assert not report.passed
        assert any("GERACAO" in issue for issue in report.issues)

    def test_skips_range_check_for_absent_column(self):
        """Se uma coluna opcional não existe, o check não deve falhar por isso."""
        df = _make_df()
        df = df.drop(columns=["TEMPERATURA"])
        report = DataQualityChecker().check(df)
        assert not any("TEMPERATURA" in issue for issue in report.issues)

    def test_multiple_issues_reported(self):
        """Vários problemas no mesmo df devem gerar vários issues."""
        df = _make_df()
        df.loc[df.index[0], "RADIACAO"] = -1.0
        df.loc[df.index[1], "UMIDADE"] = 200.0
        report = DataQualityChecker().check(df)
        assert len(report.issues) >= 2
