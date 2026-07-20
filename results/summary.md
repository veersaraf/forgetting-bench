# Forgetting-Bench A/B results

seed=0, n_turns=3000, k=5

| Metric (n=349 queries) | decay-off | decay-on | better |
|---|---|---|---|
| Stale-fact contradiction rate | 0.860 | 0.000 | lower |
| Stale context fraction | 0.474 | 0.000 | lower |
| Recall rate (current fact found) | 0.997 | 0.997 | higher |
| Answer accuracy (top-1 correct) | 0.980 | 0.983 | higher |
| Precision (correct fact / retrieved) | 0.257 | 0.199 | higher |
| Final memory size | 2651 | 1259 | lower |
| Peak memory size | 2651 | 1283 | lower |
| Final token count | 19251 | 9444 | lower |
