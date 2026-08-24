# Sentinel — results


## Run notes

All metrics below are on a temporal holdout the model never saw.

Elapsed: 453.4s


## Temporal split

| split       |      n |    t_min |    t_max |   days_span |   fraud_rate |   n_fraud |
|:------------|-------:|---------:|---------:|------------:|-------------:|----------:|
| train       | 354324 |    86400 |  8745772 |       100.2 |      0.03383 |     11988 |
| calibration |  88581 |  8745798 | 11246605 |        28.9 |      0.04036 |      3575 |
| test        | 147635 | 11246665 | 15811131 |        52.8 |      0.03454 |      5100 |


## Unseen categories at test time (fallback bucket)

| column        |   unseen_rows |
|:--------------|--------------:|
| id_31         |         13556 |
| DeviceInfo    |           977 |
| id_33         |           864 |
| id_30         |           145 |
| ProductCD     |             0 |
| card4         |             0 |
| card6         |             0 |
| M3            |             0 |
| M4            |             0 |
| M5            |             0 |
| M6            |             0 |
| P_emaildomain |             0 |
| R_emaildomain |             0 |
| M1            |             0 |
| M2            |             0 |


## Headline metrics (temporal holdout)

| metric | value |
|---|---|
| n | 147635 |
| n_fraud | 5100 |
| base_rate | 0.03454 |
| pr_auc | 0.48595 |
| roc_auc | 0.89484 |
| brier | 0.023201 |
| lift_over_base_rate_at_top_1pct | 24.79 |


## Calibration — predicted vs observed

|   bin |     n |   predicted_mean |   observed_rate |       gap |
|------:|------:|-----------------:|----------------:|----------:|
|     0 | 11782 |         0.000641 |        0.002037 | -0.001396 |
|     1 | 15299 |         0.000997 |        0.002549 | -0.001553 |
|     2 | 14326 |         0.002014 |        0.002792 | -0.000779 |
|     3 | 12781 |         0.003203 |        0.004147 | -0.000944 |
|     4 | 15489 |         0.004895 |        0.006263 | -0.001368 |
|     5 | 18645 |         0.006678 |        0.0096   | -0.002923 |
|     6 |  8700 |         0.012546 |        0.013218 | -0.000673 |
|     7 | 20480 |         0.017269 |        0.020801 | -0.003532 |
|     8 | 15195 |         0.039695 |        0.041987 | -0.002293 |
|     9 | 14938 |         0.262048 |        0.233565 |  0.028483 |


## Calibration resolution

| metric | value |
|---|---|
| distinct_calibrated_values | 536 |
| n_at_ceiling | 17 |
| max_probability | 0.999 |
| median_gap_between_levels | 1.4e-05 |


## Precision / recall / FP cost by threshold

|   threshold |   flagged |   flag_rate |   precision |   recall |   false_positives |   fp_per_1000_good |
|------------:|----------:|------------:|------------:|---------:|------------------:|-------------------:|
|        0.05 |     17500 |     0.11854 |      0.2074 |   0.7118 |             13870 |              97.31 |
|        0.1  |      9174 |     0.06214 |      0.3291 |   0.592  |              6155 |              43.18 |
|        0.2  |      5778 |     0.03914 |      0.4418 |   0.5006 |              3225 |              22.63 |
|        0.3  |      4190 |     0.02838 |      0.5391 |   0.4429 |              1931 |              13.55 |
|        0.5  |      2323 |     0.01573 |      0.7482 |   0.3408 |               585 |               4.1  |
|        0.7  |      1790 |     0.01212 |      0.8201 |   0.2878 |               322 |               2.26 |
|        0.9  |      1018 |     0.0069  |      0.888  |   0.1773 |               114 |               0.8  |


## Top features by gain

| feature   |      gain |
|:----------|----------:|
| V258      | 1281.19   |
| V201      |  853.791  |
| C7        |  316.639  |
| V156      |  239.67   |
| V294      |  206.246  |
| C4        |  151.48   |
| V147      |  143.943  |
| V70       |  141.826  |
| V254      |  138.029  |
| C8        |  136.942  |
| V91       |  133.852  |
| V225      |  112.071  |
| V324      |   98.0523 |
| C14       |   96.7594 |
| V187      |   88.8856 |
| V148      |   78.6025 |
| V57       |   73.46   |
| V98       |   66.2079 |
| V162      |   66.1602 |
| C1        |   64.8422 |


## Do our engineered velocity features earn their place?

Velocity features account for **3.58%** of total model gain. They carry real weight alongside the dataset's own C-series counters.


## Velocity features by gain

| feature              |     gain |
|:---------------------|---------:|
| DeviceInfo_amt_1d    | 27.3896  |
| addr1_amtratio_7d    | 15.5155  |
| DeviceInfo_cnt_7d    | 13.2515  |
| DeviceInfo_cnt_1d    | 11.8911  |
| card1_amt_7d         | 10.8689  |
| DeviceInfo_amt_7d    | 10.6857  |
| card1_cnt_7d         | 10.1558  |
| addr1_cnt_7d         | 10.1338  |
| addr1_amt_7d         | 10.0739  |
| addr1_amtratio_1d    |  9.95711 |
| addr1_cnt_1d         |  9.76734 |
| DeviceInfo_amt_1h    |  9.66281 |
| P_emaildomain_cnt_7d |  9.4494  |
| DeviceInfo_cnt_1h    |  9.3303  |
| card1_cnt_1d         |  9.04825 |


## Chosen operating point (fitted on calibration, evaluated on test)

| metric | value |
|---|---|
| t_review | 0.20384614169597626 |
| t_block | 0.4615384638309479 |
| n | 147635 |
| n_allow | 141857 |
| n_review | 3007 |
| n_block | 2771 |
| review_rate | 0.02037 |
| block_rate | 0.01877 |
| fraud_caught | 2488 |
| fraud_total | 5100 |
| recall_effective | 0.4878 |
| precision_block | 0.6853 |
| false_positives | 872 |
| fp_per_1000_good | 6.118 |
| policy_cost | 5569518.72 |
| do_nothing_cost | 9206043.53 |
| net_saved | 3636524.8 |
| pct_loss_prevented | 39.5 |


## Threshold generalisation

| metric | value |
|---|---|
| fitted_on | calibration slice (never sees test labels) |
| test_pct_loss_prevented | 39.5 |
| oracle_pct_loss_prevented | 39.61 |
| price_of_hindsight_pp | 0.11 |
| price_of_hindsight_rupees | 10271.71 |
| calib_pct_loss_prevented | 52.86 |
| calib_fraud_rate | 0.04036 |
| test_fraud_rate | 0.03454 |
| note | Compare test against ORACLE, not against calibration. The calibration window has a higher fraud rate, so it scores higher regardless of threshold quality. A small oracle gap means the cutoffs transfer. |


## Per-segment thresholds (fitted on calibration, by ProductCD)

| segment   |     n |   n_fraud | fitted   |   t_review |   t_block | note   |
|:----------|------:|----------:|:---------|-----------:|----------:|:-------|
| W         | 69951 |      1478 | True     |    0.15842 |   0.48162 |        |
| C         | 12187 |      1611 | True     |    0.24895 |   0.33333 |        |
| S         |  1296 |        87 | True     |    0.24895 |   0.33943 |        |
| R         |  2883 |       208 | True     |    0.31    |   0.51136 |        |
| H         |  2264 |       191 | True     |    0.29167 |   0.33943 |        |


## Per-segment operating point

| metric | value |
|---|---|
| policy | per-segment thresholds |
| n_review | 2954 |
| n_block | 3029 |
| review_rate | 0.02001 |
| block_rate | 0.02052 |
| fraud_caught | 2522 |
| recall_effective | 0.4945 |
| false_positives | 1030 |
| fp_per_1000_good | 7.226 |
| policy_cost | 5601290.12 |
| net_saved | 3604753.4 |
| pct_loss_prevented | 39.16 |


## Money: Sentinel vs baselines

| policy                                        |        cost |    net_saved | note                                                  |
|:----------------------------------------------|------------:|-------------:|:------------------------------------------------------|
| do nothing (approve all)                      | 9.20604e+06 |  0           | the loss you are trying to prevent                    |
| block everything                              | 9.97745e+07 | -9.05685e+07 | zero fraud, no business -- the degenerate upper bound |
| naive threshold 0.5                           | 6.55206e+06 |  2.65399e+06 | what a submission that skips the economics reports    |
| Sentinel (cost-optimised, 3-tier)             | 5.56952e+06 |  3.63652e+06 | review 2.04% of traffic, block 1.88%                  |
| Sentinel + per-segment thresholds (ProductCD) | 5.60129e+06 |  3.60475e+06 | recall 0.494, 7.226 FP per 1000 good                  |


## Review-capacity curve

**Review-only strategy — not comparable with the operating point above.** This curve never blocks, so it pays no friction cost; the chosen policy does both. Rows with `within_policy_cap = False` exceed the review budget the policy was fitted under and are shown for context only.


## Review-capacity curve (review-only, no blocking)

|   review_capacity |   fraud_caught |   recall |   gross_recovered |   review_spend |        net_saved |   roi_per_review |   remaining_loss | within_policy_cap   |
|------------------:|---------------:|---------:|------------------:|---------------:|-----------------:|-----------------:|-----------------:|:--------------------|
|                 0 |            0   |   0      |       0           |              0 |      0           |             0    |      9.20604e+06 | True                |
|                50 |           41.4 |   0.0081 |  100346           |           2120 |  98225.5         |          1964.51 |      9.1057e+06  | True                |
|               100 |           82.8 |   0.0162 |  184682           |           4240 | 180442           |          1804.42 |      9.02136e+06 | True                |
|               250 |          200.7 |   0.0394 |  405273           |          10810 | 394463           |          1577.85 |      8.80077e+06 | True                |
|               500 |          411.3 |   0.0806 |  772521           |          21290 | 751231           |          1502.46 |      8.43352e+06 | True                |
|              1000 |          801   |   0.1571 |       1.4388e+06  |          43300 |      1.3955e+06  |          1395.5  |      7.76725e+06 | True                |
|              2500 |         1600.2 |   0.3138 |       2.84836e+06 |         121660 |      2.7267e+06  |          1090.68 |      6.35768e+06 | True                |
|              5000 |         2180.7 |   0.4276 |       3.91262e+06 |         277310 |      3.63531e+06 |           727.06 |      5.29342e+06 | False               |
|             10000 |         2796.3 |   0.5483 |       5.07031e+06 |         606790 |      4.46352e+06 |           446.35 |      4.13574e+06 | False               |


## Feature drift, train vs test (PSI)

| feature      |    psi | status   |
|:-------------|-------:|:---------|
| hour         | 0.0224 | OK       |
| card1_cnt_1h | 0.0143 | OK       |
| amt_log      | 0.0079 | OK       |
| dist1        | 0.0076 | OK       |


## Spike sentinel — segment scan

| segment_col   | segment_value    |      n |   buckets |   mean_score_first_half |   mean_score_second_half | score_alarm   |   score_cusum_max |   fraud_rate_first_half |   fraud_rate_second_half | fraud_alarm   |
|:--------------|:-----------------|-------:|----------:|------------------------:|-------------------------:|:--------------|------------------:|------------------------:|-------------------------:|:--------------|
| ProductCD     | S                |   4018 |        53 |                 0.04252 |                  0.05389 | True          |             7.955 |                 0.06854 |                  0.0823  | True          |
| DeviceInfo    | rv:11.0          |    308 |        51 |                 0.04467 |                  0.02248 | True          |             7.216 |                 0.08505 |                  0.01434 | True          |
| ProductCD     | R                |   6634 |        53 |                 0.0587  |                  0.06811 | True          |             6.191 |                 0.04604 |                  0.04958 | False         |
| card4         | mastercard       |  48007 |        53 |                 0.03441 |                  0.0393  | True          |             6.136 |                 0.03111 |                  0.03336 | True          |
| DeviceInfo    | rv:59.0          |    256 |        49 |                 0.10697 |                  0.10375 | True          |             5.723 |                 0.08914 |                  0.12294 | True          |
| DeviceInfo    | Trident/7.0      |   1219 |        53 |                 0.02889 |                  0.02539 | True          |             5.375 |                 0.02487 |                  0.02225 | True          |
| DeviceInfo    | MacOS            |   2067 |        53 |                 0.03334 |                  0.05271 | True          |             5.298 |                 0.04    |                  0.04175 | False         |
| card4         | american express |   1391 |        53 |                 0.05773 |                  0.04282 | True          |             5.283 |                 0.04519 |                  0.02186 | True          |
| card4         | visa             |  95875 |        53 |                 0.03204 |                  0.03561 | True          |             5.212 |                 0.03323 |                  0.03619 | False         |
| card4         | discover         |   1618 |        53 |                 0.11209 |                  0.09566 | True          |             4.824 |                 0.12246 |                  0.0955  | True          |
| DeviceInfo    | Windows          |   9628 |        53 |                 0.08767 |                  0.07935 | False         |             3.933 |                 0.0818  |                  0.08532 | False         |
| card6         | debit            | 113072 |        53 |                 0.02274 |                  0.02589 | False         |             3.439 |                 0.02367 |                  0.02627 | False         |


## Spike sentinel — segments not monitored, and why

| segment_col   | segment_value      | reason                  |
|:--------------|:-------------------|:------------------------|
| DeviceInfo    | 2PYB2              | only 1 rows (need 200)  |
| DeviceInfo    | 4009F              | only 2 rows (need 200)  |
| DeviceInfo    | 4013M Build/KOT49H | only 1 rows (need 200)  |
| DeviceInfo    | 4047A Build/NRD90M | only 2 rows (need 200)  |
| DeviceInfo    | 4047G Build/NRD90M | only 3 rows (need 200)  |
| DeviceInfo    | 47418              | only 1 rows (need 200)  |
| DeviceInfo    | 5010G Build/MRA58K | only 22 rows (need 200) |
| DeviceInfo    | 5010S Build/MRA58K | only 7 rows (need 200)  |
| DeviceInfo    | 5011A Build/NRD90M | only 8 rows (need 200)  |
| DeviceInfo    | 5012G Build/MRA58K | only 4 rows (need 200)  |
| DeviceInfo    | 5015A Build/LMY47I | only 10 rows (need 200) |
| DeviceInfo    | 5025G Build/LMY47I | only 7 rows (need 200)  |


## Where it fails — recall by segment

| segment   |      n |   n_fraud |   fraud_rate |   recall |   flag_rate |
|:----------|-------:|----------:|-------------:|---------:|------------:|
| W         | 117584 |      2336 |      0.01987 |   0.2539 |      0.0229 |
| S         |   4018 |       220 |      0.05475 |   0.3955 |      0.0358 |
| H         |   4148 |       250 |      0.06027 |   0.68   |      0.0938 |
| C         |  15251 |      1985 |      0.13016 |   0.7511 |      0.1551 |
| R         |   6634 |       309 |      0.04658 |   0.7735 |      0.0594 |


## Analyst review queue (top 15 by expected loss)

|   transaction_id |   p_fraud |   amount |   expected_loss | reasons                                                                                                                                                                                |
|-----------------:|----------:|---------:|----------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|          3442600 |    0.9383 |     2681 |         4063.96 | velocity counter C1; velocity counter C13; velocity counter C11; amount is 19.6x this billing address's 7 days average; amount is 12.9x this billing address's 24 hours average        |
|          3567759 |    0.7797 |     2680 |         3375.93 | velocity counter C13; velocity counter C1; velocity counter C11; transaction hour (8:00); amount is 20.6x this billing address's 7 days average                                        |
|          3538503 |    0.6897 |     2161 |         2628.28 | Vesta risk feature V294; velocity counter C9; M3 = 1; this billing address: 6s since previous transaction; amount is 12.2x this billing address's 24 hours average                     |
|          3434782 |    0.8947 |      994 |         2365.68 | timedelta feature D2; card4 = 0; card6 = 0; amount is 8.2x this billing address's 7 days average; transaction amount Rs 994                                                            |
|          3561179 |    0.6897 |     1651 |         2276.55 | card4 = 0; P_emaildomain = 3; card6 = 0; this email domain: 14.7 h since previous transaction; amount is 9.1x this billing address's 7 days average                                    |
|          3552592 |    0.8605 |      994 |         2275.07 | timedelta feature D2; card4 = 0; velocity counter C13; transaction amount Rs 994; amount is 5.7x this billing address's 7 days average                                                 |
|          3513829 |    0.9939 |      600 |         2236.2  | velocity counter C14; Vesta risk feature V156; velocity counter C13; amount is 4.6x this email domain's 7 days average; amount is 2.9x this billing address's 7 days average           |
|          3451545 |    0.8278 |      994 |         2188.74 | card4 = 0; Vesta risk feature V294; card6 = 0; transaction amount Rs 994; amount is 6.7x this billing address's 7 days average                                                         |
|          3451548 |    0.8278 |      994 |         2188.74 | card4 = 0; Vesta risk feature V294; card6 = 0; transaction amount Rs 994; amount is 6.6x this billing address's 7 days average                                                         |
|          3565978 |    0.8278 |      994 |         2188.74 | timedelta feature D2; card4 = 0; transaction amount Rs 994; amount is 8.0x this billing address's 7 days average; velocity counter C13                                                 |
|          3556347 |    0.999  |      500 |         2147.85 | device/identity signal id_04; Vesta risk feature V156; velocity counter C14; amount is 4.3x this email domain's 7 days average; amount is 2.5x this billing address's 24 hours average |
|          3556335 |    0.999  |      500 |         2147.85 | device/identity signal id_04; Vesta risk feature V156; velocity counter C14; amount is 4.6x this email domain's 7 days average; transaction amount Rs 500                              |
|          3557572 |    0.999  |      500 |         2147.85 | device/identity signal id_04; velocity counter C14; Vesta risk feature V156; amount is 3.0x this billing address's 7 days average; amount is 3.8x this email domain's 7 days average   |
|          3556337 |    0.9939 |      500 |         2136.81 | device/identity signal id_04; Vesta risk feature V156; velocity counter C14; amount is 4.4x this email domain's 7 days average; amount is 2.5x this email domain's 24 hours average    |
|          3464150 |    0.9939 |      475 |         2111.96 | velocity counter C14; velocity counter C1; velocity counter C13; amount is 3.9x this email domain's 7 days average; transaction amount Rs 475                                          |


## Failure drill — degradation paths

| scenario   | path    |   p_fraud | decision   | degraded   |   latency_ms | note                                                                             |
|:-----------|:--------|----------:|:-----------|:-----------|-------------:|:---------------------------------------------------------------------------------|
| healthy    | model   |    0.0147 | ALLOW      | False      |        37.94 |                                                                                  |
| model_down | rules   |    0.15   | REVIEW     | True       |         0    | model unavailable: simulated model service unavailable; rules fired: no rule fir |
| slow       | timeout |    0.0147 | REVIEW     | True       |       360    | exceeded latency budget; failed safe to review                                   |
