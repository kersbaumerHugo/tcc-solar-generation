# src/data/validators/quality_checks.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from src.utils.logger import logger

# Limites físicos plausíveis por coluna.
# Tupla (min, max) — None significa "sem limite neste lado".
_RANGE_CHECKS: List[Tuple[str, Optional[float], Optional[float]]] = [
    ("RADIACAO",    0,    5_000),   # Kj/m² por hora — máximo físico terrestre
    ("TEMPERATURA", -50,  60),      # °C — extremos registrados na Terra
    ("UMIDADE",     0,    100),     # %
    ("GERACAO",     0,    None),    # MWh — nunca negativo
]

_REQUIRED_COLUMNS: Tuple[str, ...] = ("RADIACAO", "GERACAO")


@dataclass
class QualityReport:
    """
    Resultado de uma validação de qualidade de dados.

    Retornar um relatório em vez de lançar exceção dá ao chamador a
    liberdade de decidir: abortar o pipeline, registrar aviso e continuar,
    ou acumular problemas de múltiplos datasets antes de reportar.

    Atributos:
        passed: True se nenhum check falhou.
        issues: lista de mensagens descrevendo cada problema encontrado.
    """

    passed: bool = True
    issues: List[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        """Registra um problema e marca o relatório como reprovado."""
        self.passed = False
        self.issues.append(message)

    def __str__(self) -> str:
        if self.passed:
            return "Qualidade OK"
        header = f"Qualidade FALHOU ({len(self.issues)} problema(s)):"
        body = "\n".join(f"  - {i}" for i in self.issues)
        return f"{header}\n{body}"


class DataQualityChecker:
    """
    Valida um DataFrame antes de entrar no pipeline de transformação.

    Três categorias de checks:
      1. Colunas obrigatórias — se faltarem, todo o resto pode falhar.
      2. Timestamps duplicados — quebram resample() e TimeSeriesSplit.
      3. Faixas de valores — detecta erros de leitura ou unidades trocadas.

    Uso típico (em run_training.py ou ETLPipeline):
        report = DataQualityChecker().check(df)
        if not report.passed:
            logger.warning(str(report))
    """

    def check(self, df: pd.DataFrame) -> QualityReport:
        """Executa todos os checks e retorna um QualityReport consolidado."""
        report = QualityReport()
        self._check_required_columns(df, report)
        self._check_no_duplicate_timestamps(df, report)
        self._check_value_ranges(df, report)

        if report.passed:
            logger.debug("Validação de qualidade: OK")
        else:
            logger.warning(f"Validação de qualidade: {len(report.issues)} problema(s)")

        return report

    @staticmethod
    def _check_required_columns(df: pd.DataFrame, report: QualityReport) -> None:
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            report.fail(f"Colunas obrigatórias ausentes: {missing}")

    @staticmethod
    def _check_no_duplicate_timestamps(df: pd.DataFrame, report: QualityReport) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            return
        n_dupes = int(df.index.duplicated().sum())
        if n_dupes > 0:
            report.fail(f"{n_dupes} timestamp(s) duplicado(s) no índice")

    @staticmethod
    def _check_value_ranges(df: pd.DataFrame, report: QualityReport) -> None:
        for col, low, high in _RANGE_CHECKS:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if low is not None:
                n = int((series < low).sum())
                if n > 0:
                    report.fail(f"{col}: {n} valor(es) abaixo do mínimo esperado ({low})")
            if high is not None:
                n = int((series > high).sum())
                if n > 0:
                    report.fail(f"{col}: {n} valor(es) acima do máximo esperado ({high})")
