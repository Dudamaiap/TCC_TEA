| modelo | família | AUC média (folds) | AUC dp (folds) | AUC OOF | IC95 inf | IC95 sup | Acc. balanceada | Sensibilidade | Especificidade | F1 | IC inferior > 0,5 | execuções com 1 só classe prevista |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SVM_RBF | Clássico | 0.724 | 0.049 | 0.712 | 0.664 | 0.749 | 0.663 | 0.598 | 0.728 | 0.625 | sim | 0 |
| XGBoost | Clássico | 0.691 | 0.022 | 0.678 | 0.655 | 0.701 | 0.653 | 0.558 | 0.748 | 0.603 | sim | 0 |
| Random_Forest | Clássico | 0.683 | 0.036 | 0.653 | 0.616 | 0.689 | 0.614 | 0.410 | 0.818 | 0.490 | sim | 0 |
| GCN | Grafo | 0.640 | 0.032 | 0.615 | 0.583 | 0.646 | 0.600 | 0.600 | 0.599 | 0.577 | sim | 0 |
| GAT | Grafo | 0.639 | 0.054 | 0.590 | 0.556 | 0.624 | 0.586 | 0.575 | 0.596 | 0.549 | sim | 0 |
| ResNet50 | Deep learning | 0.552 | 0.041 | 0.525 | 0.491 | 0.574 | 0.543 | 0.510 | 0.575 | 0.502 | não | 0 |
| Dummy_prior | Referência | 0.500 | 0.000 | — | — | — | 0.500 | 0.000 | 1.000 | 0.000 | — | 5 |
| CNN | Deep learning | 0.493 | 0.017 | 0.491 | 0.454 | 0.530 | 0.504 | 0.577 | 0.430 | 0.508 | não | 0 |
| Dummy_estratificado | Referência | 0.463 | 0.031 | — | — | — | 0.463 | 0.382 | 0.544 | 0.402 | — | 0 |

Na base TEA, o melhor desempenho médio entre folds foi do modelo SVM_RBF (AUC = 0.724 ± 0.049; AUC OOF = 0.712, IC 95% por bootstrap de sites [0.664, 0.749]). O limite inferior do IC ficou acima de 0,5, o que atende ao critério definido a priori; isso não constitui prova de efeito nem de utilidade clínica. Acurácia balanceada média de 0.663, sensibilidade 0.598 e especificidade 0.728. Os resultados vêm de validação cruzada de 5 folds agrupada por site, não de um conjunto de teste independente; não foram comparados com o diagnóstico clínico.
