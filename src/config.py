"""Configuração central do projeto.

Todos os notebooks importam daqui:

    from src.config import BASE_DIR, PATHS, SEED, criar_pastas

Os caminhos são derivados da localização deste arquivo (PROJECT_ROOT), portanto
não dependem da máquina, do sistema operacional nem do diretório de execução.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_DIR = PROJECT_ROOT

DATA_DIR = BASE_DIR / "Data"

# Mesmas chaves usadas nos notebooks originais (Colab). Valores agora são pathlib.Path.
PATHS = {
    "raw":            DATA_DIR / "raw",
    "roi_timeseries": DATA_DIR / "roi_timeseries",
    "conectivity":    DATA_DIR / "conectivity",   # grafia original preservada
    "processed":      DATA_DIR / "processed",
    "figures":        BASE_DIR / "Figures",
    "models":         BASE_DIR / "Models",
    "results":        BASE_DIR / "Results",
    "docs":           BASE_DIR / "Docs",
}

# Semente global (metodologia original)
SEED = 42


def criar_pastas(verbose: bool = False) -> None:
    """Cria as pastas do projeto, se ainda não existirem."""
    for nome, caminho in PATHS.items():
        caminho.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"OK  [{nome}] -> {caminho}")
