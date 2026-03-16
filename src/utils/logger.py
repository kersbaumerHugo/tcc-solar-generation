# src/utils/logger.py
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(
    name: str = "solar_generation",
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Configura e retorna um logger configurado.
    
    Args:
        name: Nome do logger
        log_level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Se True, salva logs em arquivo
        log_dir: Diretório para salvar os arquivos de log
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Evita duplicação de handlers
    if logger.handlers:
        return logger
    
    # Formato das mensagens
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para arquivo (opcional)
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"solar_generation_{timestamp}.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Instância padrão para importação rápida
logger = setup_logger()