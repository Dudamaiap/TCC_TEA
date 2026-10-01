"""Funções utilitárias compartilhadas (infraestrutura; não alteram a metodologia)."""
import os
import random
from pathlib import Path

import numpy as np


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


def definir_sementes(seed: int = 42, tensorflow: bool = False) -> None:
    """Fixa as sementes de random e NumPy (e, opcionalmente, TensorFlow/Keras).

    scikit-learn e XGBoost recebem a semente via `random_state` nos próprios
    estimadores/divisores, exatamente como nos notebooks originais.
    """
    random.seed(seed)
    np.random.seed(seed)
    if tensorflow:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)  # também fixa random e NumPy


def detectar_dispositivo_xgboost(verbose: bool = True) -> str:
    """Retorna 'cuda' se o XGBoost conseguir usar uma GPU NVIDIA; senão 'cpu'.

    Os notebooks originais fixavam device='cuda' (GPU T4 do Colab). Em máquinas
    sem GPU isso gera erro; aqui fazemos uma sondagem mínima e caímos para CPU.
    """
    try:
        from xgboost import XGBClassifier
        sonda = XGBClassifier(n_estimators=1, tree_method="hist", device="cuda", n_jobs=1)
        sonda.fit(np.random.RandomState(0).rand(20, 4), np.array([0, 1] * 10))
        dispositivo = "cuda"
    except Exception:
        dispositivo = "cpu"
    if verbose:
        if dispositivo == "cuda":
            print("GPU detectada — XGBoost usará device='cuda'.")
        else:
            print("GPU não detectada — XGBoost executando em CPU (device='cpu').")
    return dispositivo


def info_gpu_tensorflow(verbose: bool = True):
    """Lista GPUs visíveis ao TensorFlow, sem quebrar se não houver."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if verbose:
        if gpus:
            print("GPU detectada:", gpus)
        else:
            print("GPU não detectada — executando em CPU")
    return gpus
