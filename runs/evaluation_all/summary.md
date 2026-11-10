# Comparaison des modeles
Sources : runs/evaluation, runs/evaluation_v2, runs/evaluation_equi

## accuracy

| modele | val | test_1_2A | test_2_3A | test_3_4A |
|---|---|---|---|---|
| cnn_e50 | 0.738 | 0.722 | 0.407 | 0.281 |
| equi_origine | 0.078 | 0.073 | 0.064 | 0.053 |
| cnn_e100 | 0.757 | 0.739 | 0.410 | 0.281 |
| equi_r4_n512 | 0.323 | 0.298 | 0.206 | 0.145 |

## f1_macro

| modele | val | test_1_2A | test_2_3A | test_3_4A |
|---|---|---|---|---|
| cnn_e50 | 0.710 | 0.697 | 0.391 | 0.255 |
| equi_origine | 0.062 | 0.058 | 0.046 | 0.039 |
| cnn_e100 | 0.732 | 0.716 | 0.396 | 0.256 |
| equi_r4_n512 | 0.264 | 0.246 | 0.153 | 0.081 |

## auc_macro

| modele | val | test_1_2A | test_2_3A | test_3_4A |
|---|---|---|---|---|
| cnn_e50 | 0.981 | 0.976 | 0.878 | 0.801 |
| equi_origine | 0.601 | 0.568 | 0.569 | 0.549 |
| cnn_e100 | 0.983 | 0.978 | 0.877 | 0.797 |
| equi_r4_n512 | 0.833 | 0.813 | 0.737 | 0.647 |

## ap_macro

| modele | val | test_1_2A | test_2_3A | test_3_4A |
|---|---|---|---|---|
| cnn_e50 | 0.784 | 0.773 | 0.410 | 0.252 |
| equi_origine | 0.074 | 0.064 | 0.062 | 0.059 |
| cnn_e100 | 0.803 | 0.790 | 0.415 | 0.253 |
| equi_r4_n512 | 0.274 | 0.249 | 0.161 | 0.104 |

## top3_acc

| modele | val | test_1_2A | test_2_3A | test_3_4A |
|---|---|---|---|---|
| cnn_e50 | 0.938 | 0.909 | 0.670 | 0.549 |
| equi_origine | 0.225 | 0.201 | 0.188 | 0.176 |
| cnn_e100 | 0.942 | 0.913 | 0.673 | 0.547 |
| equi_r4_n512 | 0.604 | 0.568 | 0.432 | 0.329 |
