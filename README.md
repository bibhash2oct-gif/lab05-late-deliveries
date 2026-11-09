# Late Deliveries

QuickBite wants to know, when an order is placed, whether it will arrive late, so it
can offer a $4 voucher first. This project predicts the chance that an order is late
and flags it when that chance is at or above a threshold chosen from the company's costs.

## Setup
~~~
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install -r requirements.txt
pip install -e .
~~~

## Run
~~~
python -m latedeliveries.train
~~~

## Check it
~~~
python -m ruff format --check .
python -m ruff check .
python -m pytest
~~~

## Decisions

### Columns
Used, because each is known when the order is placed:
- Numeric: `distance_km`, `items`, `hour`, `prep_minutes`, `courier_months`
- Category: `weather`, `day_type`, `vehicle`

Left out:
- `customer_rating`: the customer gives it after the order arrives, so it does not
  exist at prediction time. It is also strongly tied to lateness. Adding it back
  raised CV ROC-AUC from 0.795 to 0.907. That jump is leakage, not a better model.
- `order_id`: a label for the order. It carries no information about the delivery.
- `late`: the target.

### Model
Five-fold stratified CV on the 2,250 training orders only (mean ± std):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Best CV cost |
|---|---|---|---|---|---|---|
| dummy (most frequent) | 0.844 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | $5,265 |
| logistic | 0.862 ± 0.009 | 0.654 ± 0.065 | 0.242 ± 0.053 | 0.352 ± 0.064 | 0.795 ± 0.034 | $4,031 |
| logistic, balanced | 0.737 ± 0.019 | 0.333 ± 0.027 | 0.681 ± 0.049 | 0.447 ± 0.034 | 0.795 ± 0.034 | $4,098 |
| random forest, balanced | 0.787 ± 0.017 | 0.382 ± 0.034 | 0.584 ± 0.058 | 0.461 ± 0.040 | 0.782 ± 0.025 | $4,237 |
| gradient boosting | 0.861 ± 0.007 | 0.635 ± 0.050 | 0.265 ± 0.024 | 0.373 ± 0.029 | 0.788 ± 0.031 | $4,152 |

"Best CV cost" is the total cost on the out-of-fold predictions at each model's own
best threshold. Flagging nobody costs $5,265 on the same orders.

I chose plain logistic regression:
- The company pays in dollars, so I compared models by cost, not by accuracy or F1 at
  0.5. Precision, recall and F1 change with the threshold, and I choose the threshold
  later from costs.
- Logistic regression has the best ranking (ROC-AUC 0.795) and the lowest cost ($4,031).
  The gaps to the other models are smaller than the fold spread (about ±0.03), so
  the tie-breaker is that it is the simplest and easiest to explain.
- Without class weights its probabilities stay close to real chances, so the best
  threshold lands near the $4 / $15 break-even point (see Threshold).

### Tuning
GridSearchCV over `model__C` [0.01, 0.1, 1, 10] and `model__class_weight`
[None, balanced]: 8 combinations, 5 folds each. Scoring is ROC-AUC, because the
threshold is chosen afterwards from costs, so tuning should improve how well the model
ranks orders, not its score at one fixed cut-off.

Best settings: `C=10`, `class_weight=None`, CV ROC-AUC 0.795. The untuned model scored
0.795 ± 0.034. Tuning did **not** beat the untuned model by more than the fold-to-fold
spread; all 8 settings were within 0.003 of each other.

### Threshold
A flagged order costs $4. A late order that is not flagged costs $15. So flagging pays
when the chance of being late is above 4 / 15 ≈ 0.27. The default 0.5 would miss most
late orders. I took out-of-fold probabilities on the training data with
`cross_val_predict`, costed every threshold from 0.05 to 0.95, and picked the cheapest:
**0.30** ($4,035, against $5,265 for flagging nobody). The test set played no part.

## Result
Test set, 750 orders (117 late), scored once at threshold 0.30:

|  | Predicted on time | Predicted late |
|---|---|---|
| Actually on time | 578 | 55 |
| Actually late | 64 | 53 |

Precision 0.491, recall 0.453, F1 0.471, ROC-AUC 0.796.

| Policy | Cost on the test set |
|---|---|
| Flag nobody (today) | $1,755 |
| Flag everybody | $3,000 |
| Use the model | $1,392 |

The model saves $363 (21%) against today's policy. That is close to the 23% it saved
in cross-validation, so I trust that the model beats flagging nobody, but the exact
saving would move by a few hundred dollars on a different set of 750 orders.

## What the tests cover
- The feature list leaves out `customer_rating`, `order_id` and `late` (parametrized).
- The split is stratified (same late rate in both halves) and repeatable.
- Predicted probabilities are between 0 and 1 and sum to 1.
- The pipeline predicts for an order with a missing `prep_minutes` and an unseen
  weather value (`hail`).
- The cost function and threshold choice on small examples worked out on paper,
  including one where flagging nobody is cheapest and one where flagging everybody is.
- A saved pipeline predicts the same after reloading (uses `tmp_path`).

Not tested: the exact numbers from a full `train.py` run. That run takes several
seconds and its numbers depend on the scikit-learn version, so I check it by running
it from a fresh clone and comparing `reports/metrics.json` instead.

## Known limitations
- `prep_minutes` is the actual kitchen time. At the moment an order is placed only an
  estimate exists, so a real system would need the restaurant's estimated prep time,
  and the saving should be checked again with it.
- `hour` is treated as a number, so logistic regression cannot fully capture the two
  rush periods (12–13 and 18–20). As a category it lifts CV ROC-AUC to about 0.81,
  which is within the fold spread, but it is worth trying.
- The data is invented. The costs ($4, $15) are fixed; if they change, the threshold
  must be chosen again.
