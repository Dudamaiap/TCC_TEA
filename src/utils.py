"""Funções utilitárias compartilhadas (infraestrutura; não alteram a metodologia)."""
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np


def _gpu_nvidia_disponivel() -> bool:
    """Retorna True se houver uma GPU NVIDIA visível para o sistema."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        resultado = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return resultado.returncode == 0 and bool(resultado.stdout.strip())


def exigir_arquivo(caminho, notebook_origem: str) -> Path:
    """Garante que um arquivo produzido por outro notebook exista.

    Levanta FileNotFoundError com mensagem indicando qual notebook o gera.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo {caminho.name} não encontrado.\n"
            f"Caminho esperado: {caminho}\n\n"
            f"Este arquivo é produzido pelo notebook:\n{notebook_origem}\n\n"
            f"Execute esse notebook antes de continuar."
        )
    return caminho


def definir_sementes(seed: int = 42) -> None:
    """Fixa as sementes de random e NumPy.

    PyTorch recebe a semente nos notebooks de treino; scikit-learn e XGBoost
    recebem `random_state` nos estimadores e divisores.
    """
    random.seed(seed)
    np.random.seed(seed)


def detectar_dispositivo_xgboost(verbose: bool = True) -> str:
    """Retorna 'cuda' se houver GPU NVIDIA visível; senão 'cpu'.

    A checagem é feita com indicadores do sistema/driver e não depende do
    comportamento silencioso de fallback do XGBoost em máquinas sem GPU.
    """
    gpu_ok = _gpu_nvidia_disponivel()
    if not gpu_ok:
        try:
            import torch
            gpu_ok = torch.cuda.is_available()
        except Exception:
            gpu_ok = False

    dispositivo = "cuda" if gpu_ok else "cpu"
    if verbose:
        if dispositivo == "cuda":
            print("GPU detectada — XGBoost usará device='cuda'.")
        else:
            print("GPU não detectada — XGBoost executando em CPU (device='cpu').")
    return dispositivo


