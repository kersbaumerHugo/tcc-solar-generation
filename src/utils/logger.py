# src/utils/logger.py
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "solar_generation",
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs",
) -> logging.Logger:
    """
    Configura e retorna um logger.

    Args:
        name: Nome do logger (namespace no módulo logging do Python)
        log_level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Se True, salva logs em arquivo com timestamp
        log_dir: Diretório para os arquivos de log
    """
    instance = logging.getLogger(name)
    instance.setLevel(getattr(logging, log_level.upper()))

    # Evita duplicação de handlers em reimportações
    if instance.handlers:
        return instance

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    instance.addHandler(console_handler)

    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"solar_generation_{timestamp}.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        instance.addHandler(file_handler)

    return instance


class _LazyLogger:
    """
    Proxy que inicializa o logger na primeira chamada, não na importação.

    Problema original: `logger = setup_logger()` no nível do módulo criava um
    arquivo de log toda vez que qualquer módulo era importado — inclusive
    durante os testes. Com o proxy, o logger (e seu arquivo) só é criado quando
    alguém realmente chama logger.info(), logger.debug(), etc.

    A API de uso permanece idêntica:
        from src.utils.logger import logger
        logger.info("mensagem")
    """

    _instance: Optional[logging.Logger] = None

    def _get(self) -> logging.Logger:
        if self._instance is None:
            _LazyLogger._instance = setup_logger()
        return self._instance

    def __getattr__(self, name: str):
        """
        Delega qualquer atributo/método para o logger real.

        Cobre debug, info, warning, error, critical, exception, log, e
        quaisquer métodos adicionados em versões futuras do módulo logging.
        """
        return getattr(self._get(), name)


logger = _LazyLogger()
