# Flow matching : effet des options

RMSD moyen (A) sur chaque ensemble ; colonnes suivantes sur **test_3_4A**. Meilleur de chaque colonne en gras. ± : ecart-type entre seeds (RMSD, squelette, < 1 A) ; les autres colonnes viennent de la seed analysee.

| modele | seeds | RMSD val | RMSD 1-2 A | RMSD 2-3 A | RMSD 3-4 A | squelette | < 1 A | echecs | orientation | GLY N/C inv. | acides D | chi1 juste | clashes | err. C-N (A) | omega < 150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Ordre de generation (sans contexte)* | | | | | | | | | | | | | | | |
| joint | 3 | 0.567 ± 0.006 | 0.673 ± 0.003 | 1.222 ± 0.012 | 1.634 ± 0.012 | 1.222 ± 0.010 | 34.2 % ± 0.5 | 66.2 % | 41.5 % | 7.4 % | 0.8 % | 48.4 % | **0.9 %** | 1.016 | 42.2 % |
| atome | 1 | 0.592 | 0.717 | 1.378 | 1.792 | 1.339 | 32.4 % | 67.6 % | 39.8 % | 12.9 % | **0.0 %** | 54.4 % | 2.8 % | 1.126 | 46.6 % |
| backbone | 3 | 0.520 ± 0.003 | 0.632 ± 0.007 | 1.217 ± 0.002 | 1.671 ± 0.009 | 1.199 ± 0.005 | 35.2 % ± 0.1 | 64.8 % | 36.8 % | 9.4 % | 11.8 % | 52.5 % | 2.6 % | 0.984 | 39.8 % |
| hybride | 3 | 0.517 ± 0.002 | 0.622 ± 0.006 | 1.208 ± 0.012 | 1.655 ± 0.008 | 1.200 ± 0.006 | 35.7 % ± 0.2 | 64.5 % | 36.1 % | 9.7 % | 12.9 % | 52.2 % | 2.8 % | 0.988 | 39.6 % |
| backbone_nco | 1 | 0.540 | 0.647 | 1.262 | 1.714 | 1.225 | 34.9 % | 65.1 % | 37.4 % | 9.5 % | **0.0 %** | 56.6 % | 2.6 % | 1.009 | 39.7 % |
| hybrid_nco | 1 | 0.523 | 0.642 | 1.267 | 1.724 | 1.222 | 35.2 % | 64.8 % | 37.9 % | 9.6 % | **0.0 %** | 55.3 % | 2.1 % | 0.990 | 40.1 % |
| *Contexte C-alpha et loss peptidique* | | | | | | | | | | | | | | | |
| hybrid_pep | 1 | 0.514 | 0.625 | 1.203 | 1.644 | 1.188 | 36.3 % | 63.7 % | 35.2 % | 10.0 % | 11.6 % | 51.9 % | 2.0 % | 0.961 | 39.2 % |
| hybrid_knn_pep | 1 | 0.454 | 0.531 | 0.814 | 1.054 | 0.499 | 57.8 % | 42.2 % | 5.6 % | 7.8 % | 2.5 % | 66.9 % | 1.1 % | 0.261 | 10.5 % |
| hybrid_chain_pep | 1 | **0.442** | **0.515** | **0.788** | 0.951 | **0.388** | 60.6 % | 39.4 % | **4.2 %** | **0.0 %** | 0.6 % | **68.6 %** | 2.2 % | **0.067** | **4.2 %** |
| hybrid_chain_pep_chi | 1 | 0.463 | 0.536 | 0.795 | **0.948** | 0.389 | **61.1 %** | **38.9 %** | **4.2 %** | **0.0 %** | 0.6 % | 68.4 % | 1.5 % | 0.069 | 4.4 % |
| hybrid_chain_pep_chi3 | 1 | 0.489 | 0.561 | 0.823 | 0.976 | 0.417 | 59.6 % | 40.4 % | 4.8 % | **0.0 %** | 0.5 % | 67.8 % | 1.4 % | 0.083 | 5.4 % |

Sources : runs/evaluation_flow_chi_e100/summary.csv, runs/evaluation_flow_context_e100/summary.csv, runs/evaluation_flow_e100/summary.csv, runs/evaluation_flow_nco_e100/summary.csv, runs/seeds_flow_e100/seeds_summary.csv, runs/analysis_flow_chi_e100, runs/analysis_flow_context_e100, runs/analysis_flow_e100, runs/analysis_flow_nco_e100, runs/geometry_chi_e100, runs/geometry_context_e100, runs/geometry_nco_e100
