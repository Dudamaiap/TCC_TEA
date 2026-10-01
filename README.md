# TCC_TEA_TDAH — Classificação de TEA com Machine Learning e Deep Learning (ABIDE I)

Classificação de Transtorno do Espectro Autista (TEA × Controle) a partir de conectividade funcional
(rs-fMRI, atlas CC200, correlação de Pearson) do **ABIDE I**, comparando SVM, Random Forest, XGBoost,
uma CNN customizada e a ResNet50 (Transfer Learning), com validação `StratifiedGroupKFold` (5 folds,
grupos = `SITE_ID`) e explicabilidade via SHAP.

Este projeto foi migrado do Google Colab + Google Drive para execução **local no VS Code**.
A metodologia não foi alterada (ver `RELATORIO_MIGRACAO.md`).

## Estrutura

```text
TCC_TEA_TDAH/
├── notebooks/        # 01 … 11 (executar em ordem)
├── src/
│   ├── config.py     # caminhos (pathlib), PATHS, SEED — importado por todos os notebooks
│   └── utils.py      # exigir_arquivo, sementes, detecção de GPU
├── Data/
│   ├── raw/              # ABIDE_pcp/, phenotypic_abide.csv, cc200_roi_atlas.nii.gz
│   ├── roi_timeseries/   # roi_timeseries_abide.pkl
│   ├── conectivity/      # matrizes de conectividade (.pkl)   (grafia "conectivity" preservada)
│   └── processed/        # dataset_abide_processado.pkl (X, y, groups, sub_ids, folds)
├── Figures/          # .png/.pdf
├── Models/           # .joblib e .keras
├── Results/          # .csv e .pkl de resultados
├── Docs/             # relatórios (ex.: exclusões), resumos
├── .vscode/settings.json
├── requirements.txt
├── environment.yml
└── RELATORIO_MIGRACAO.md
```

## Requisitos

- **Python 3.12 ou 3.13** (versões testadas na origem: Python 3.13)
- **VS Code** com as extensões *Python* e *Jupyter*
- ~**30 GB livres** para o download completo do ABIDE I (derivativo `rois_cc200`, 871 sujeitos; o download original levou ~22 min)
- GPU NVIDIA é **opcional** (tudo roda em CPU)

## Instalação

Abra a pasta `TCC_TEA_TDAH/` no VS Code e, no terminal integrado:

```bash
python -m venv .venv
```

Ativar o ambiente:

```bash
# Windows (PowerShell / cmd)
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Se o PowerShell bloquear a ativação: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Instalar as dependências:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Selecionar o interpretador no VS Code

1. `Ctrl+Shift+P` → **Python: Select Interpreter** → escolha `.venv` (`.venv\Scripts\python.exe`).
2. Abra `notebooks/01_Configuracao.ipynb` → canto superior direito **Select Kernel** → *Python Environments* → `.venv`.

O arquivo `.vscode/settings.json` define `jupyter.notebookFileRoot` como a raiz do projeto, e cada notebook
também localiza `src/` automaticamente (funciona com o diretório de trabalho em `notebooks/` ou na raiz).

## Execução (ordem obrigatória)

```text
01 Configuração → 02 Download ABIDE → 03 Exploração → 04 Conectividade → 05 Validação
→ 06 Análise Exploratória → 07 Preparação dos Dados → 08 Baselines
→ 09 Deep Learning → 10 XAI → 11 Resultados Finais
```

Cada notebook começa com **Objetivo / Entradas / Processamento / Saídas**. Se um arquivo de entrada
não existir, o notebook para com uma mensagem informando qual notebook o produz.

### Reaproveitando dados já gerados no Colab/Drive

Copie as pastas de `MyDrive/TCC_TEA_TDAH/TEA/` para a raiz do projeto, mantendo os nomes
(`Data/`, `Models/`, `Results/`, `Figures/`, `Docs/`). O notebook 02 detecta
`phenotypic_abide.csv` + `roi_timeseries_abide.pkl` e **não baixa de novo**. O atlas CC200 e os demais
downloads também só ocorrem se o arquivo não existir.

## GPU

Verifique (opcional):

```bash
nvidia-smi
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
python -c "import xgboost; print(xgboost.__version__)"
```

- O notebook **01** e o **09** imprimem `GPU detectada` ou `GPU não detectada — executando em CPU`.
- **XGBoost (notebook 08):** o original fixava `device='cuda'`. Agora o notebook testa se a GPU funciona e,
  senão, usa `device='cpu'`. A mesma configuração (`tree_method='hist'`) é mantida, mas **GPU e CPU não
  são bit-a-bit idênticas**: os resultados do XGBoost em CPU podem diferir ligeiramente dos registrados.
  Em CPU o grid do XGBoost é mais lento (`n_jobs=1` foi preservado).
- **TensorFlow no Windows:** a partir do TensorFlow 2.11 **não há suporte a GPU no Windows nativo**.
  Opções: (a) rodar em CPU (padrão, funciona); (b) usar **WSL2 + Ubuntu** com driver NVIDIA do Windows e
  `pip install "tensorflow[and-cuda]"` (CUDA vem junto via pip, não é necessário instalar o CUDA Toolkit
  manualmente). Este projeto **não instala CUDA automaticamente**.
- Em GPU o notebook 09 ativa `mixed_float16` (comportamento original); em CPU mantém `float32`.
- Treinar a CNN e a ResNet50 por 5 folds em CPU pode levar muitas horas.

## Dados

Todos os dados ficam em `Data/` (ignorado pelo Git). O ABIDE I é obtido por `nilearn.datasets.fetch_abide_pcp`
com os parâmetros originais (`pipeline="cpac"`, `band_pass_filtering=True`, `global_signal_regression=False`,
`derivatives=["rois_cc200"]`, `quality_checked=True`). O primeiro download exige internet; o ResNet50 também
baixa os pesos ImageNet na primeira execução (cache do Keras em `~/.keras`).

## Resultados

| O quê | Onde |
|---|---|
| Figuras | `Figures/` (`03_*`, `04_*`, `06_*`, `08_*`, `10_*`, `11_*`) |
| Modelos | `Models/` (`08_modelos_baselines.joblib`, `09_*.keras`) |
| Tabelas/resultados | `Results/` (`08_resultados_baselines.csv`, `09_comparacao_final_todos_modelos.csv`, `11_tabela_resultados_finais.csv`, `06_*.pkl`) |
| Documentos | `Docs/` (`03_resumo_exploracao.txt`, `05_relatorio_exclusoes.csv` se houver exclusões) |

## Compatibilidade de versões (importante)

- **pandas × pickle:** os `.pkl` originais foram gravados com **pandas 3.x**. Abrir `dataset_abide_processado.pkl`
  com pandas 2.x gera `TypeError: StringDtype.__init__() takes from 1 to 2 positional arguments but 3 were given`
  (isso ocorreu no notebook 09 original). Por isso `requirements.txt` fixa `pandas==3.0.6`. O notebook 08
  mantém o tratamento original desse erro (reconstrução do dataset) — leia `RELATORIO_MIGRACAO.md` sobre
  limitações desse fallback.
- **NumPy:** `numpy==2.1.3` (versão do original) é compatível com o TensorFlow ≥ 2.18; não suba para NumPy ≥ 2.2 sem testar.
- **xgboost 3.4.x** exige Python ≥ 3.12; **pandas 3.x** exige Python ≥ 3.11.
- **SHAP × Keras 3:** o `shap.GradientExplainer` funcionou na verificação da migração com SHAP 0.52.0,
  TensorFlow 2.21 e Keras 3.15 (modelo de teste em dados sintéticos). SHAP/TensorFlow muito antigos podem não funcionar.
- **scikit-learn 1.6.1** é a versão com a qual os modelos `.joblib` foram gerados; carregá-los com outra
  versão pode emitir avisos ou falhar.
- Os pesos de `.keras` salvos em GPU com `mixed_float16` carregam normalmente em CPU.

## Reprodutibilidade

`SEED = 42` (`src/config.py`): `random`, `NumPy`, `StratifiedGroupKFold(random_state=42)`, `random_state`
dos estimadores do scikit-learn/XGBoost. No notebook 09 foi **adicionada** a semente do TensorFlow
(`tf.keras.utils.set_random_seed`); o original não a fixava. Operações determinísticas do TF **não** foram
ativadas; treinos em GPU podem variar levemente.
