# Relatório da migração Google Colab → VS Code local

## 1. Arquivos modificados (os 11 notebooks, em `notebooks/`)

| Notebook | Alterações (todas de infraestrutura, salvo as marcadas ⚠) |
|---|---|
| 01 | Reescrito como configuração local: seeds, criação de pastas, versões, GPU. Sem `drive.mount` / `!pip`. |
| 02 | Config central; `import pickle`/`pandas` adicionados ⚠(ver §4-3); download **pulado** se `phenotypic_abide.csv` e `roi_timeseries_abide.pkl` já existem; parâmetros de `fetch_abide_pcp` idênticos. |
| 03 | Config central; bloco inline de `PATHS` removido; `data_dir=str(PATHS['raw'])`; verificação de entradas. |
| 04, 05, 06, 07 | Config central; `!pip` removido; verificação de entradas com mensagem indicando o notebook produtor. |
| 08 | Idem; XGBoost: `device='cuda'` → detecção automática (`cuda` se funcionar, senão `cpu`); gráficos das células com valores hardcoded agora também salvam PNG em `Figures/`; essas células foram **marcadas** como valores previamente registrados. Tratamento `StringDtype` mantido. |
| 09 | Removida a célula de `pip install --upgrade pandas` + aviso de reiniciar runtime; `nvidia-smi` (shell) → verificação em Python; mensagens `GPU detectada` / `GPU não detectada — executando em CPU`; semente do TensorFlow adicionada ⚠(§4-10); aspas escapadas corrigidas ⚠(§4-1). |
| 10 | Config central; `pip` removido; `edge_threshold` corrigido ⚠(§4-2). |
| 11 | Estava **vazio**. Agora consolida `Results/09_comparacao_final_todos_modelos.csv` em tabela e gráfico; seções sem código original marcadas **SEÇÃO NÃO IMPLEMENTADA NO NOTEBOOK ORIGINAL**. |

Todos os notebooks ganharam um cabeçalho **Objetivo / Entradas / Processamento / Saídas**.
Saídas (resultados, gráficos) dos notebooks originais foram **preservadas**, exceto as de células puramente Colab/pip/nvidia-smi e o traceback de `StringDtype` do 09 (erro de ambiente do Colab). Nas saídas mantidas, o caminho `/content/drive/MyDrive/TCC_TEA_TDAH/TEA` foi substituído por `<BASE_DIR>`.

## 2. Arquivos criados

`src/__init__.py`, `src/config.py`, `src/utils.py`, `requirements.txt`, `environment.yml`, `.gitignore`, `README.md`, `RELATORIO_MIGRACAO.md`, `.vscode/settings.json`, `.gitkeep` nas pastas de dados/saída.

## 3. Alterações realizadas

```text
Google Drive → sistema de arquivos local (pathlib)
/content/drive/MyDrive/... → PROJECT_ROOT (src/config.py)
!pip install → requirements.txt
google.colab / drive.mount → removidos
PATHS/BASE_DIR repetidos em cada notebook → from src.config import BASE_DIR, PATHS, SEED
nvidia-smi → verificação via Python / TensorFlow
XGBoost device='cuda' fixo → cuda se disponível, senão cpu
```

## 4. PROBLEMAS IDENTIFICADOS

**Corrigidos (necessários para rodar localmente):**

1. **09 · células da CNN e da ResNet50 (última linha de impressão)** — `print(f\"Accuracy média: ...\")` com `\"` literal ⇒ `SyntaxError`; as células de treino não executavam. Impacto: bloqueava o 09. **Corrigido** (`\"` → `"`), sem mudança de lógica.
2. **10 · célula do grafo (`plot_connectome`)** — `edge_threshold=f'{limiar_shap}'` passa string numérica sem `%`; nilearn 0.14.1 levanta `ValueError`. O notebook 10 original **não tinha nenhuma saída registrada**, ou seja, não há evidência de que tenha rodado. **Corrigido** para `float(limiar_shap)` (mesma intenção: limiar = percentil 98 de |SHAP|, top 2%).
3. **02 · células de salvamento** — usavam `pd` e `pickle` sem importar (dependiam de estado do kernel do Colab); a célula 1 do arquivo fornecido também tinha `except Exception as e:` sem corpo (SyntaxError). **Corrigido** com os imports; o bloco inteiro foi substituído pela configuração local.

**Mantidos como no original — decisão sua:**

4. **08 · fallback `StringDtype` (célula 2)**: usa `read_csv(index_col=0)` e `sub_ids` como `str`. O 07 usa `set_index('SUB_ID')` e `sub_ids` inteiros. Se o fallback for acionado, o `.loc[sub_ids]` pode falhar e, se funcionar, `sub_ids` vira `str` e o 09 (`conn_dict[s]` com chaves inteiras) quebraria. Sugestão: no 07, gravar `groups` com `.astype(str).to_numpy()` e evitar o fallback.
5. **07 · origem do erro `StringDtype`**: `phenotypic.loc[..., 'SITE_ID'].values` com pandas 3.x grava dtype do pandas dentro do `.pkl`, ilegível em pandas 2.x. Por isso `pandas==3.0.6` foi fixado.
6. **08 · `GridSearchCV(cv=3)` interno** não é agrupado por site e o `StandardScaler` é ajustado antes dele: a escolha de hiperparâmetros pode ser otimista (há mistura de sites nos folds internos). Os folds externos continuam separados por site, então a estimativa de teste não é afetada por isso.
7. **08 · célula do gráfico por fold** sobrescreve a variável `folds` com `[0,1,2,3,4]`. Hoje não afeta nada (é a última a usá-la), mas reexecutar o treino depois dela quebraria.
8. **09 · `validation_split=0.15`** do Keras separa os **últimos** 15% do treino, sem embaralhar e sem agrupar por site; a validação (usada por EarlyStopping/ReduceLROnPlateau) tende a concentrar os últimos sites da ordem de `sub_ids`. Limitação metodológica a documentar no TCC.
9. **09 · ResNet50** recebe entrada em [0, 1] sem `preprocess_input` (a rede foi treinada com 0–255 e subtração de média). O texto do notebook sugere compatibilidade; vale revisar a justificativa.
10. **09 · reprodutibilidade**: o original não fixava semente do TensorFlow. Foi **adicionado** `tf.keras.utils.set_random_seed(SEED)` (não ativa determinismo de operações). Isso é uma mudança de ambiente, não de arquitetura/hiperparâmetros, mas os resultados do 09 não são comparáveis bit-a-bit aos anteriores (que já não eram reproduzíveis).
11. **08 · XGBoost em CPU** pode diferir ligeiramente do registrado (executado em GPU). Documentado no README.
12. **08 · valores hardcoded** (acurácia por fold e matrizes de confusão agregadas): conferi com as saídas originais — SVM e RF coincidem com as saídas impressas, e XGBoost coincide com as saídas do fold (acc 0,756/0,595/0,618/0,633/0,631 e matrizes). Não recalculados.
13. **10** usa apenas o fold 0 (41 sujeitos de teste) para o SHAP do XGBoost e 10 sujeitos para a CNN; os resultados de explicabilidade refletem esse subconjunto.

## 5. Problemas corrigidos

Somente os itens 1, 2 e 3 (e os imports/caminhos descritos acima).

## 6. NÃO foi alterado

Dataset e parâmetros do `fetch_abide_pcp`; `SEED = 42`; `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` com `groups = SITE_ID`; `y_bin = (y == 1).astype(int)` (0 = Controle, 1 = TEA); `ConnectivityMeasure(kind='correlation')` e vetorização 200×199/2 = 19.900; `StandardScaler` ajustado dentro de cada fold (sem leakage); SVM/RF/XGBoost, grades de hiperparâmetros, `GridSearchCV(cv=3, scoring='f1')`; métricas (Accuracy, F1, AUC-ROC); CNN customizada, ResNet50, EarlyStopping, ReduceLROnPlateau, batch/épocas; SHAP (`TreeExplainer`, `GradientExplainer`); nomes de todos os arquivos de entrada e saída.

## 7. Dependências

Ver `requirements.txt` (pandas 3.0.6, numpy 2.1.3, scikit-learn 1.6.1, nilearn 0.14.1, nibabel 5.4.2, xgboost 3.4.1, scipy 1.16.3, joblib 1.6.0 — as versões observadas nas saídas originais; matplotlib, seaborn, shap, tensorflow e ipykernel com versão mínima, pois as saídas originais não as registram).

## 8. Como executar

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```
VS Code: abrir a pasta `TCC_TEA_TDAH/`, selecionar o interpretador `.venv` e o kernel, executar `notebooks/01` … `11` em ordem.

## 9. O que foi efetivamente testado

Ambiente de teste: Linux, Python 3.13, **mesmas versões** do original (numpy/pandas/sklearn/nilearn/xgboost/…), TensorFlow 2.21, SHAP 0.52, **sem GPU**. Como os dados reais do ABIDE não foram fornecidos, os testes usaram **dados sintéticos** (84 sujeitos, 7 sites, 200 ROIs, atlas sintético) — validam caminhos, E/S, sintaxe e o encadeamento entre notebooks, **não** os resultados científicos.

| Notebook | Resultado |
|---|---|
| 01 | Executou completo. |
| 02 | Executou completo no caminho "dados já existem → download pulado". **O download real não foi testado** (sem acesso às fontes do ABIDE). |
| 03 | Testadas só as células offline (leitura do CSV, gráficos diagnóstico×site e idade/sexo). **Não validado**: download do dado bruto, `func_mean`, atlas CC200 real e figuras de neuroimagem (exigem internet). |
| 04, 05, 06, 07 | Executaram completos. |
| 08 | Executou completo com **grades de hiperparâmetros reduzidas** na cópia de teste (o notebook entregue mantém as grades originais); XGBoost caiu para CPU como previsto; salvou CSV, `.joblib` e figuras. A execução com as grades originais não foi concluída no ambiente de teste (2 CPUs). |
| 09 | Testados: verificação de GPU/TF, carregamento dos dados, arquitetura da CNN e treino da CNN (4 dos 5 folds antes de ser interrompido por tempo; salvou os `.keras`). **Não validado**: ResNet50 (download dos pesos ImageNet) e a célula final de consolidação. A célula com a f-string corrigida passa na verificação de sintaxe, mas não foi executada. |
| 10 | Executou completo (XGBoost + SHAP, ROI, grafo, CNN + `GradientExplainer`) com os modelos sintéticos de teste. |
| 11 | Executou completo sobre um CSV de teste. |

**Não foi possível validar com os arquivos fornecidos:** reprodução dos resultados originais (os `.pkl`, `.joblib`, `.keras` e CSVs reais não foram enviados), download do ABIDE, ResNet50, uso de GPU e GPU no Windows.
