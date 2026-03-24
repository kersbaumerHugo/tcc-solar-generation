"""
Pipeline de pré-processamento (ETL).

Orquestra a transformação dos dados brutos (Excel da ANEEL + CSVs INMET)
em arquivos *_full.csv prontos para o run_training.py.

Estrutura esperada dos dados de entrada:
    data/raw/geracao/     → arquivos .xlsx de geração (ANEEL)
    data/raw/climatico/   → subpastas por ano com CSVs do INMET

Uso:
    python scripts/run_preprocessing.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import Config
from src.data.extractors.climate import ClimateDataExtractor
from src.data.extractors.generation import SolarGenerationExtractor
from src.data.pipeline import ETLPipeline
from src.utils.logger import logger


def check_inputs() -> bool:
    """Verifica diretórios de entrada antes de executar o ETL."""
    required = {
        "Geração (ANEEL)": Config.RAW_GERACAO_DIR,
        "Clima (INMET)": Config.RAW_CLIMA_DIR,
    }
    missing = [(label, path) for label, path in required.items() if not path.exists()]
    if missing:
        for label, path in missing:
            logger.error(f"Diretório não encontrado — {label}: {path}")
        return False
    return True


def main() -> None:
    logger.info("=" * 60)
    logger.info("PIPELINE DE PRÉ-PROCESSAMENTO (ETL)")
    logger.info("=" * 60)

    if not check_inputs():
        logger.error(
            "\nEstrutura esperada:\n"
            f"  {Config.RAW_GERACAO_DIR}\n"
            f"  {Config.RAW_CLIMA_DIR}\n"
        )
        return

    gen_extractor = SolarGenerationExtractor(Config.RAW_GERACAO_DIR)
    clim_extractor = ClimateDataExtractor(Config.RAW_CLIMA_DIR)

    pipeline = ETLPipeline(
        gen_extractor=gen_extractor,
        clim_extractor=clim_extractor,
        output_dir=Config.FULL_DIR,
    )

    summary = pipeline.run()

    if summary.empty:
        logger.warning("Nenhuma usina processada com sucesso.")
    else:
        logger.info(f"\n{summary.to_string()}")
        logger.info(f"\nDatasets prontos em: {Config.FULL_DIR}")


if __name__ == "__main__":
    main()
