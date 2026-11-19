| modelo | família | AUC média (folds) | AUC dp (folds) | AUC OOF | IC95 inf | IC95 sup | Acc. balanceada | Sensibilidade | Especificidade | F1 | IC inferior > 0,5 | execuções com 1 só classe prevista |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SVM_RBF | Clássico | 0.7241 | 0.0494 | 0.7121 | 0.664 | 0.7486 | 0.6629 | 0.5981 | 0.7277 | 0.6251 | sim | 0 |
| XGBoost | Clássico | 0.6907 | 0.0224 | 0.6781 | 0.6546 | 0.7013 | 0.6531 | 0.558 | 0.7482 | 0.603 | sim | 0 |
| Random_Forest | Clássico | 0.6833 | 0.0364 | 0.6532 | 0.616 | 0.6891 | 0.614 | 0.4104 | 0.8176 | 0.49 | sim | 0 |
| GCN | Grafo | 0.6395 | 0.0319 | 0.6151 | 0.5828 | 0.6455 | 0.5998 | 0.6003 | 0.5992 | 0.5766 | sim | 0 |
| GAT | Grafo | 0.639 | 0.0535 | 0.59 | 0.5562 | 0.6243 | 0.5857 | 0.5749 | 0.5965 | 0.5493 | sim | 0 |
| Dummy_prior | Referência | 0.5 | 0.0 |  |  |  | 0.5 | 0.0 | 1.0 | 0.0 | — | 5 |
| Dummy_estratificado | Referência | 0.4631 | 0.0307 |  |  |  | 0.4631 | 0.382 | 0.5441 | 0.4024 | — | 0 |

Na base TEA, o melhor desempenho médio entre folds continua sendo do SVM-RBF (AUC = 0.724 ± 0.049; AUC OOF = 0.712; IC 95% por bootstrap de sites [0.664, 0.749]). O limite inferior acima de 0,5 atende ao critério definido a priori, sem provar efeito ou utilidade clínica. A avaliação usa validação cruzada de cinco folds agrupada por site, não um conjunto de teste independente.
