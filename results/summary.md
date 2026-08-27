# Forgetting-Bench results

seed=0, n_turns=3000, k=5, tau=150

| Metric (n=367 queries) | keep-everything | last-write-wins | ebbinghaus-decay | learned-forget | want |
|---|---|---|---|---|---|
| Stale-fact contradiction rate | 0.807 | 0.330 | 0.319 | 0.322 | lower |
| Stale context fraction | 0.359 | 0.066 | 0.064 | 0.064 | lower |
| Recall rate (current fact found) | 0.760 | 0.670 | 0.668 | 0.668 | higher |
| Answer accuracy (top-1 correct) | 0.657 | 0.657 | 0.654 | 0.657 | higher |
| Precision (correct fact / retrieved) | 0.190 | 0.134 | 0.134 | 0.134 | higher |
| Final memory size | 2633 | 2325 | 1361 | 198 | lower |
| Final token count | 18859 | 17121 | 9903 | 1346 | lower |

## Live incumbent probes

- mem0: not runnable (mem0ai is not installed)
- letta: not runnable (letta-client (or letta) is not installed)

Incumbents not scored:
- mem0 is not runnable: mem0ai is not installed
- letta is not runnable: letta-client (or letta) is not installed
