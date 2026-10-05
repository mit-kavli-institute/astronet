# 4-class vs P-vs-notP — comparison results

Arms: `/pdo/astronet-data/models/vetting/experimental/pablomer/binary_pvsnotp_nopretrain/20260806/4class` vs `/pdo/astronet-data/models/vetting/experimental/pablomer/binary_pvsnotp_nopretrain/20260806/binary` (5/5 members)

## In-distribution test split (dec2025_cad_scat_v5_aug)

Thresholds are each arm's F1-optimum on the **val** split, frozen before touching test/S93. AP/ROC and @thr metrics use the across-seed ensemble-mean score.

| arm | N | N_pos | AP | ROC-AUC | thr | P@thr | R@thr | F1@thr |
|---|---|---|---|---|---|---|---|---|
| 4class | 1548 | 582 | 0.9933 | 0.9953 | 0.653 | 0.959 | 0.959 | 0.959 |
| binary | 1548 | 582 | 0.9934 | 0.9954 | 0.592 | 0.962 | 0.960 | 0.961 |
| 4class @0.215 (prod) | 1548 | 582 | - | - | 0.215 | 0.875 | 0.990 | 0.929 |

**Per-seed (member) metrics:**

| arm | member | AP | ROC-AUC |
|---|---|---|---|
| 4class | AstroCNNModelVetting_pablomer_final_nopretrain_20260806_113052 | 0.9915 | 0.9943 |
| 4class | AstroCNNModelVetting_pablomer_final_nopretrain_20260806_115703 | 0.9926 | 0.9950 |
| 4class | AstroCNNModelVetting_pablomer_final_nopretrain_20260806_122116 | 0.9921 | 0.9945 |
| 4class | AstroCNNModelVetting_pablomer_final_nopretrain_20260806_124641 | 0.9916 | 0.9941 |
| 4class | AstroCNNModelVetting_pablomer_final_nopretrain_20260806_131318 | 0.9926 | 0.9947 |
| **4class mean ± std** | 5 seeds | 0.9921 ± 0.0005 | 0.9945 ± 0.0003 |
| binary | AstroCNNModelVetting_pablomer_final_binary_nopretrain_20260806_114512 | 0.9925 | 0.9949 |
| binary | AstroCNNModelVetting_pablomer_final_binary_nopretrain_20260806_120927 | 0.9919 | 0.9949 |
| binary | AstroCNNModelVetting_pablomer_final_binary_nopretrain_20260806_123412 | 0.9920 | 0.9945 |
| binary | AstroCNNModelVetting_pablomer_final_binary_nopretrain_20260806_130040 | 0.9903 | 0.9934 |
| binary | AstroCNNModelVetting_pablomer_final_binary_nopretrain_20260806_132643 | 0.9920 | 0.9944 |
| **binary mean ± std** | 5 seeds | 0.9918 ± 0.0008 | 0.9944 ± 0.0005 |

**Precision at matched recall** (best P with R ≥ target):

| arm | P@R≥0.6 | P@R≥0.7 | P@R≥0.8 | P@R≥0.9 |
|---|---|---|---|---|
| 4class | 1.000 | 0.998 | 0.996 | 0.991 |
| binary | 1.000 | 1.000 | 0.996 | 0.987 |

**Recall at matched precision** (best R with P ≥ target):

| arm | R@P≥0.6 | R@P≥0.7 | R@P≥0.8 | R@P≥0.9 |
|---|---|---|---|---|
| 4class | 0.998 | 0.998 | 0.993 | 0.985 |
| binary | 0.998 | 0.997 | 0.997 | 0.990 |

## Sector 93 (Aug-5 vetting sheets)

Scores from sector-93-scatter tfrecords; labels joined on astro_id (= TIC*100 + planetno). Caveat: the labelled set is the union of astronet top-322 and operator picks — a preselected slice, so these are comparative numbers between arms, not sector-wide absolutes.

### Policy: unanimous

| arm | N | N_pos | unscored | AP | ROC-AUC | thr | P@thr | R@thr | F1@thr |
|---|---|---|---|---|---|---|---|---|---|
| 4class | 279 | 95 | 4 | 0.7084 | 0.7729 | 0.653 | 0.363 | 0.989 | 0.531 |
| binary | 279 | 95 | 4 | 0.6876 | 0.7551 | 0.592 | 0.357 | 0.968 | 0.521 |

### Policy: any_planet

| arm | N | N_pos | unscored | AP | ROC-AUC | thr | P@thr | R@thr | F1@thr |
|---|---|---|---|---|---|---|---|---|---|
| 4class | 370 | 130 | 4 | 0.6506 | 0.7190 | 0.653 | 0.365 | 0.954 | 0.528 |
| binary | 370 | 130 | 4 | 0.6252 | 0.7012 | 0.592 | 0.364 | 0.954 | 0.527 |

### Policy: strict_planet

| arm | N | N_pos | unscored | AP | ROC-AUC | thr | P@thr | R@thr | F1@thr |
|---|---|---|---|---|---|---|---|---|---|
| 4class | 370 | 95 | 4 | 0.6420 | 0.7680 | 0.653 | 0.276 | 0.989 | 0.432 |
| binary | 370 | 95 | 4 | 0.6131 | 0.7482 | 0.592 | 0.270 | 0.968 | 0.422 |

**Per-seed (member) AP, unanimous policy:**

| arm | member | AP |
|---|---|---|
| 4class | AstroCNNModelVetting_pablomer_final_nopretrain_20260806_113052 | 0.6738 |
| 4class | AstroCNNModelVetting_pablomer_final_nopretrain_20260806_115703 | 0.7027 |
| 4class | AstroCNNModelVetting_pablomer_final_nopretrain_20260806_122116 | 0.6749 |
| 4class | AstroCNNModelVetting_pablomer_final_nopretrain_20260806_124641 | 0.7029 |
| 4class | AstroCNNModelVetting_pablomer_final_nopretrain_20260806_131318 | 0.6850 |
| **4class mean ± std** | 5 seeds | 0.6879 ± 0.0128 |
| binary | AstroCNNModelVetting_pablomer_final_binary_nopretrain_20260806_114512 | 0.6939 |
| binary | AstroCNNModelVetting_pablomer_final_binary_nopretrain_20260806_120927 | 0.6848 |
| binary | AstroCNNModelVetting_pablomer_final_binary_nopretrain_20260806_123412 | 0.6767 |
| binary | AstroCNNModelVetting_pablomer_final_binary_nopretrain_20260806_130040 | 0.6553 |
| binary | AstroCNNModelVetting_pablomer_final_binary_nopretrain_20260806_132643 | 0.6591 |
| **binary mean ± std** | 5 seeds | 0.6740 ± 0.0148 |

