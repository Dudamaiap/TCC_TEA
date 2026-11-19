# TCC_TEA_TDAH — Classificação de TEA com Machine Learning e Deep Learning (ABIDE I)

Classificação de Transtorno do Espectro Autista (TEA × Controle) a partir de conectividade funcional
(rs-fMRI, atlas CC200, correlação de Pearson) do **ABIDE I**, comparando SVM, Random Forest, XGBoost, MLP e BrainNetCNN; nos grafos, GCN, GAT e BrainGNN, com validação `StratifiedGroupKFold` (5 folds,
grupos = `SITE_ID`) e explicabilidade via SHAP.

Este projeto foi migrado do Google Colab + Google Drive para execução **local no VS Code**.
O fluxo atual usa MLP e BrainNetCNN no deep learning e GCN, GAT e BrainGNN nos grafos.

## Estrutura

```text
TCC_TEA_TDAH/
├── notebooks/        # fluxo ABIDE; 09 MLP/BrainNetCNN; 10 GCN/GAT/BrainGNN
├── src/
│   ├── config.py     # caminhos (pathlib), PATHS, SEED — importado por todos os notebooks
│   └── utils.py      # exigir_arquivo, sementes, detecção de GPU
├── Data/
│   ├── raw/              # ABIDE_pcp/, phenotypic_abide.csv, cc200_roi_atlas.nii.gz
│   ├── roi_timeseries/   # roi_timeseries_abide.pkl
│   ├── conectivity/      # matrizes de conectividade (.pkl)   (grafia "conectivity" preservada)
│   └── processed/        # dataset_abide_processado.pkl (X, y, groups, sub_ids, folds)
├── Figures/          # .png/.pdf
├── Models/           # pesos dos modelos de grafos e dos novos modelos
├── Results/          # .csv e .pkl de resultados
├── Docs/             # relatórios (ex.: exclusões), resumos
├── .vscode/settings.json
├── requirements.txt
├── environment.yml
└── RELATORIO_MIGRACAO.md
```

## Requisitos

- **Python 3.12** (os notebooks locais usam o kernel `.venv` com Python 3.12)
- **VS Code** com as extensões *Python* e *Jupyter*
- ~**30 GB livres** para o download completo do ABIDE I (derivativo `rois_cc200`, 871 sujeitos; o download original levou ~22 min)
- GPU NVIDIA é **opcional** (tudo roda em CPU)

## Instalação

Abra a pasta `TCC_TEA_TDAH/` no VS Code e, no terminal integrado:

```bash
# Windows
py -3.12 -m venv .venv

# Linux / macOS
python3.12 -m venv .venv
```

Se `.venv` já tiver sido criado com outra versão do Python, recrie-o com um dos comandos acima antes de instalar as dependências.

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

1. `Ctrl+Shift+P` → **Python: Select Interpreter** → escolha o ambiente `.venv`.
2. Abra `notebooks/01_Configuracao.ipynb` → canto superior direito **Select Kernel** → *Python Environments* → `.venv`.

O arquivo `.vscode/settings.json` define `jupyter.notebookFileRoot` como a raiz do projeto, e cada notebook
também localiza `src/` automaticamente (funciona com o diretório de trabalho em `notebooks/` ou na raiz).

## Execução (ordem obrigatória)

```text
01 Configuração → 02 Download ABIDE → 03 Exploração → 04 Conectividade → 05 Validação
→ 06 Análise Exploratória → 07 Preparação dos Dados → 08 Baselines
→ 09 MLP/BrainNetCNN → 10 GCN/GAT/BrainGNN → 11 XAI → 12 Resultados Finais
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
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import xgboost; print(xgboost.__version__)"
```

- Os notebooks de deep learning e grafos usam GPU se disponível e continuam executáveis em CPU.
- **XGBoost (notebook 08):** o original fixava `device='cuda'`. Agora o notebook testa se a GPU funciona e,
  senão, usa `device='cpu'`. A mesma configuração (`tree_method='hist'`) é mantida, mas **GPU e CPU não
  são bit-a-bit idênticas**: os resultados do XGBoost em CPU podem diferir ligeiramente dos registrados.
  Em CPU o grid do XGBoost é mais lento (`n_jobs=1` foi preservado).
- Os notebooks atuais de deep learning e grafos usam MLP/BrainNetCNN e GCN/GAT/BrainGNN.

## Dados

Todos os dados ficam em `Data/` (ignorado pelo Git). O ABIDE I é obtido por `nilearn.datasets.fetch_abide_pcp`
com os parâmetros originais (`pipeline="cpac"`, `band_pass_filtering=True`, `global_signal_regression=False`, `derivatives=["rois_cc200"]`, `quality_checked=True`). O primeiro download exige internet.

## Resultados

| O quê | Onde |
|---|---|
| Figuras | `Figures/` (`03_*`, `04_*`, `06_*`, `08_*`, `10_*`, `11_*`) |
| Modelos | `Models/grafos_folds/`; novos pesos são gravados em `Models/modelos_dl_tea/` e `experimentos/08_grafos_tea/Models/` |
| Tabelas/resultados | métricas existentes em `Results/`; novos modelos gravam em `Results/modelos_dl_tea/` e `experimentos/08_grafos_tea/Results/` |
| Documentos | `Docs/` (`03_resumo_exploracao.txt`, `05_relatorio_exclusoes.csv` se houver exclusões) |

## Compatibilidade de versões (importante)

- **pandas × pickle:** os `.pkl` originais foram gravados com **pandas 3.x**. Abrir `dataset_abide_processado.pkl`
  com pandas 2.x gera `TypeError: StringDtype.__init__() takes from 1 to 2 positional arguments but 3 were given`
  (isso ocorreu no notebook 09 original). Por isso `requirements.txt` fixa `pandas==3.0.6`. O notebook 08 mantém o tratamento desse erro (reconstrução do dataset); confira suas notas de execução.
- **NumPy:** `numpy==2.1.3` é a versão fixada para os artefatos do projeto; não suba para NumPy ≥ 2.2 sem testar.
- **xgboost 3.4.x** exige Python ≥ 3.12; **pandas 3.x** exige Python ≥ 3.11.
- **scikit-learn 1.6.1** é a versão com a qual os modelos `.joblib` foram gerados; carregá-los com outra
  versão pode emitir avisos ou falhar.

## Reprodutibilidade

`SEED = 42` (`src/config.py`): `random`, `NumPy`, `StratifiedGroupKFold(random_state=42)`, `random_state`
dos estimadores do scikit-learn/XGBoost. Os notebooks atuais fixam sementes NumPy/PyTorch e os estimadores clássicos. Treinos em GPU podem variar levemente.

## Experimentos experimentais (estrutura adicionada em 2026-11)

A pasta `Data/` é compartilhada pelos notebooks. Os resultados remanescentes do fluxo principal continuam guardados. O fluxo atual usa `notebooks/09_Modelo_Deep_Learning.ipynb` para MLP/BrainNetCNN e `notebooks/10_Modelos_Grafos_ABIDE.ipynb` para GCN/GAT/BrainGNN; as novas saídas ficam isoladas em `Results/modelos_dl_tea/` e `experimentos/08_grafos_tea/`.

A auditoria local confirmou 871 participantes, 19.900 atributos Fisher Z, 468 controles, 403 participantes TEA, 20 `SITE_ID` e cinco folds externos registrados no pickle. Os folds têm tamanhos desiguais (41, 173, 165, 324 e 168); todos contêm as duas classes. O pickle não inclui `fold_id` diretamente: o módulo o reconstrói da lista `folds`.

Os arquivos e resultados de CNN e ResNet50 foram removidos. Os resultados restantes de baselines e GCN/GAT foram preservados. Ainda não há resultados de validação cruzada TEA para MLP, BrainNetCNN ou BrainGNN; as novas execuções ficam isoladas para não sobrescrever os resultados preservados.
