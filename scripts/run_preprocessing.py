"""
Pipeline de pré-processamento (ETL).

Orquestra a transformação dos dados brutos (Excel de geração + CSVs INMET)
em arquivos *_full.csv prontos para o run_training.py.

Status: os extratores do código legado ainda não foram refatorados para a
arquitetura src/data/extractors/. Este script serve como ponto de entrada
documentado enquanto a refatoração dos extratores não é concluída.

Uso (quando os extratores estiverem prontos):
    python scripts/run_preprocessing.py

TODO (próxima refatoração):
    - Refatorar gerador_csv_geracao.py → src/data/extractors/generation.py
    - Refatorar gerador_dados_climaticos.py → src/data/extractors/climate.py
    - Refatorar merge_geracao_clima.py → src/data/pipeline.py
    - Substituir os imports legados abaixo pelos novos extratores
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import Config
from src.utils.logger import logger

# Legado — será substituído pelos extratores refatorados
LEGACY_SCRIPTS = [
    Config.BASE_DIR / "gerador_csv_geracao.py",
    Config.BASE_DIR / "gerador_dados_climaticos.py",
    Config.BASE_DIR / "merge_geracao_clima.py",
]


def check_inputs() -> bool:
    """Verifica se os diretórios de entrada existem antes de processar."""
    required = [Config.GERACAO_DIR, Config.CLIMA_DIR]
    missing = [d for d in required if not d.exists()]

    if missing:
        for d in missing:
            logger.error(f"Diretório de entrada não encontrado: {d}")
        return False

    logger.info("Diretórios de entrada validados.")
    return True


def main() -> None:
    logger.info("=" * 60)
    logger.info("PIPELINE DE PRÉ-PROCESSAMENTO (ETL)")
    logger.info("=" * 60)
    logger.warning(
        "Este script ainda usa o código legado. "
        "Os extratores src/data/extractors/ precisam ser implementados."
    )

    if not check_inputs():
        logger.error("Pré-condições não atendidas. Abortando.")
        return

    logger.info(
        "\nPróximos passos para completar esta refatoração:\n"
        "  1. Implementar src/data/extractors/generation.py\n"
        "  2. Implementar src/data/extractors/climate.py\n"
        "  3. Implementar src/data/pipeline.py (ETLPipeline)\n"
        "  4. Substituir os scripts legados por essas classes\n"
        "\nReferência: CLAUDE.md → Arquitetura Alvo"
    )


if __name__ == "__main__":
    main()
