# 4-class vs P-vs-notP — comparison results

Arms: `/pdo/astronet-data/models/vetting/experimental/pablomer/binary_pvsnotp/20260805/4class` vs `/pdo/astronet-data/models/vetting/experimental/pablomer/binary_pvsnotp/20260805/binary` (1/1 members)

## In-distribution test split (dec2025_cad_scat_v5_aug)

Thresholds are each arm's F1-optimum on the **val** split, frozen before touching test/S93.

| arm | N | N_pos | AP | ROC-AUC | thr | P@thr | R@thr | F1@thr |
|---|---|---|---|---|---|---|---|---|
| 4class | 1548 | 582 | 0.9930 | 0.9950 | 0.671 | 0.966 | 0.964 | 0.965 |
| binary | 1548 | 582 | 0.9911 | 0.9941 | 0.753 | 0.975 | 0.948 | 0.962 |
| 4class @0.215 (prod) | 1548 | 582 | - | - | 0.215 | 0.916 | 0.979 | 0.947 |

**Precision at matched recall** (best P with R ≥ target):

| arm | P@R≥0.6 | P@R≥0.7 | P@R≥0.8 | P@R≥0.9 |
|---|---|---|---|---|
| 4class | 1.000 | 0.998 | 0.996 | 0.991 |
| binary | 0.995 | 0.995 | 0.994 | 0.987 |

**Recall at matched precision** (best R with P ≥ target):

| arm | R@P≥0.6 | R@P≥0.7 | R@P≥0.8 | R@P≥0.9 |
|---|---|---|---|---|
| 4class | 0.998 | 0.998 | 0.991 | 0.983 |
| binary | 0.998 | 0.998 | 0.991 | 0.983 |

## Sector 93 (Aug-5 vetting sheets)

Scores from sector-93-scatter tfrecords; labels joined on astro_id (= TIC*100 + planetno). Caveat: the labelled set is the union of astronet top-322 and operator picks — a preselected slice, so these are comparative numbers between arms, not sector-wide absolutes.

### Policy: unanimous

| arm | N | N_pos | unscored | AP | ROC-AUC | thr | P@thr | R@thr | F1@thr |
|---|---|---|---|---|---|---|---|---|---|
| 4class | 279 | 95 | 4 | 0.6389 | 0.7293 | 0.671 | 0.357 | 1.000 | 0.526 |
| binary | 279 | 95 | 4 | 0.6233 | 0.7345 | 0.753 | 0.367 | 1.000 | 0.537 |

### Policy: any_planet

| arm | N | N_pos | unscored | AP | ROC-AUC | thr | P@thr | R@thr | F1@thr |
|---|---|---|---|---|---|---|---|---|---|
| 4class | 370 | 130 | 4 | 0.5706 | 0.6615 | 0.671 | 0.361 | 0.977 | 0.527 |
| binary | 370 | 130 | 4 | 0.5552 | 0.6676 | 0.753 | 0.367 | 0.969 | 0.533 |

### Policy: strict_planet

| arm | N | N_pos | unscored | AP | ROC-AUC | thr | P@thr | R@thr | F1@thr |
|---|---|---|---|---|---|---|---|---|---|
| 4class | 370 | 95 | 4 | 0.5721 | 0.7345 | 0.671 | 0.270 | 1.000 | 0.425 |
| binary | 370 | 95 | 4 | 0.5361 | 0.7284 | 0.753 | 0.277 | 1.000 | 0.434 |

