# Forgetting-Bench

*A benchmark for whether agent memory **forgets** — not just whether it recalls.*

Mainstream agent-memory libraries (mem0, Letta/MemGPT, Zep) are built and benchmarked around **recall**: can it fetch the right fact back. The under-measured half of long-horizon memory is whether an agent **forgets stale facts, resolves contradictions, and stays bounded** over thousands of turns — because that is where agents rot: the retrieved context fills with outdated, contradictory memories and token cost grows without limit.

Forgetting-Bench measures that axis on a reproducible long-horizon workload, and ships a small reference memory core with a pluggable decay module so you can see where forgetting helps, where it costs you, and where no memory policy can win.

## The setup (why the numbers aren't gamed)

A contradiction is only measurable if you know the ground truth. So the workload is **seeded and synthetic**: 3,000 turns of facts, value updates (contradictions), and distractor noise, with the correct current value of every slot known at every turn.

The catch that makes this honest: the memory core does **not** receive clean slot keys. It **extracts** `(entity, attribute)` from raw text with a keyword extractor — and **40% of updates are phrased to defeat it** (paraphrases and implicit/numeric updates like *"Alice relocated to Denver"* or *"Bob got a raise to 110k"*, which never restate the attribute). When extraction misses, the update isn't linked to its slot, so contradiction suppression **cannot fire** and the stale fact survives. The metric, by contrast, is scored against ground truth the memory never sees. That gap is deliberate — it's what stops the contradiction metric from being a tautology, and it gives every method a real error floor.

## The finding

Three memory policies over the same workload (seed 0, 3,000 turns, 367 queries, top-5 retrieval):

| Metric (n=367 queries) | keep-everything | last-write-wins | ebbinghaus-decay | want |
|---|---|---|---|---|
| **Stale-fact contradiction rate** | 0.807 | 0.330 | **0.319** | lower |
| Stale fraction of retrieved context | 0.359 | 0.066 | **0.064** | lower |
| Recall (current fact retrieved) | 0.760 | 0.670 | 0.668 | higher |
| Answer accuracy (top-1 correct) | 0.657 | 0.657 | 0.654 | higher |
| Precision (correct fact / retrieved) | 0.190 | 0.134 | 0.134 | higher |
| **Final memory size** (entries) | 2,633 | 2,325 | **1,361** | lower |
| **Final token count** | 18,859 | 17,121 | **9,903** | lower |

Read this honestly — there are three distinct findings, not one flashy one:

1. **No policy drives contradictions to zero.** Keyword extraction misses ~40% of the updates, so there is a real contradiction *floor* (~0.33) that both dedup and decay hit. A benchmark where the number went to 0.000 would be measuring its own plumbing, not memory quality.

2. **Decay's genuine win over the mem0-style bar is bounded memory, not contradiction.** `last-write-wins` (per-slot dedup — what mem0-style fact memory already does) matches decay on contradictions and recall. But it never forgets what it can't slot, so **distractor noise accumulates forever**. Ebbinghaus decay matches its contradiction handling (0.319 vs 0.330) and recall (0.668 vs 0.670) while holding **~40% less memory / ~42% fewer tokens** — and the gap widens with horizon:

   ![Memory growth](results/bloat.png)

3. **Forgetting is a tunable frontier, not a free lunch.** Sweeping decay stability `τ` trades recall for fewer contradictions: `τ=30` cuts contradictions to 0.10 but drops recall to 0.50; `τ=1000` keeps recall at 0.67 but lets contradictions climb back to the 0.33 floor. `last-write-wins` is a *single fixed point* on this frontier; decay lets you move along it — at a fraction of the memory.

   ![Forgetting-vs-recall frontier](results/frontier.png)

Every number here is produced by `make bench` — see [Reproduce it](#reproduce-the-finding).

**On the one adverse number:** keep-everything "wins" precision (0.190 vs 0.134) and recall (0.760). That is hoarding, not skill — retaining every past value means old entries whose value coincidentally equals the current one get counted as correct hits. The same hoarding is why it has by far the *worst* contradiction rate. Reported here rather than dropped.

## Scope of the claim

This compares against a faithful **re-implementation** of last-write-wins fact memory, not a live mem0/Letta/Zep process. The adapter seam and a labeled stub exist ([below](#comparing-against-real-libraries)); wiring a real backend is the honest next step, not a shipped result. No claim here depends on running an incumbent.

## Install

```bash
git clone https://github.com/veersaraf/forgetting-bench
cd forgetting-bench
pip install -e ".[dev]"     # core is pure-stdlib; matplotlib only for plots
```

## Quickstart — the mechanism, and its blind spot

```python
from forgetting_bench.memory import MemoryStore, EbbinghausDecay
from forgetting_bench.workload import default_extractor

mem = MemoryStore(decay=EbbinghausDecay(tau=150), slot_extractor=default_extractor())
mem.add("Alice's home city is boston.", turn=0)
mem.add("Alice's home city is now denver.", turn=50)   # canonical: extractor links it

for s in mem.retrieve("What is Alice's home city?", now=60, k=2):
    print(f"{s.score:.2f}  strength={s.strength:.2f}  {s.entry.text}")
```

```
0.79  strength=1.00  Alice's home city is now denver.
0.04  strength=0.05  Alice's home city is boston.        <- contradicted, suppressed
```

Now phrase the update as a paraphrase the keyword extractor can't slot:

```python
mem.add("Alice relocated to denver.", turn=50)   # extractor misses -> no supersession
```

```
0.79  strength=1.00  Alice's home city is boston.        <- stale fact survives, ranks #1
0.49  strength=1.00  Alice relocated to denver.
```

That miss is not a bug to hide — it is exactly the failure mode Forgetting-Bench quantifies.

## Reproduce the finding

```bash
make bench      # == python -m forgetting_bench.experiment
```

Runs all three arms plus the `τ` sweep over one seeded workload, prints the table, and writes `results/summary.{json,md}` and three plots (`bloat.png`, `contradiction.png`, `frontier.png`). Deterministic: same seed, same numbers.

```bash
make test       # 44 tests: scoring, decay, supersession, extraction, metrics,
                # workload ground truth, and a guard on the finding across seeds
```

## How it works

```
forgetting_bench/
├── memory/           the reference core
│   ├── store.py        generative-agents retrieval: recency + importance + relevance
│   ├── decay.py        pluggable policy: NoDecay, LastWriteWins, EbbinghausDecay
│   ├── extractor.py    imperfect keyword slot-extraction (the honest crux)
│   ├── embedder.py     dependency-free sparse hashing embedder (the "relevance" term)
│   ├── importance.py   heuristic importance scorer (LLM-swappable)
│   └── entry.py        the memory stream's atomic entry
├── workload/         seeded synthetic long-horizon stream with ground truth
├── bench/            harness + the forgetting-axis metrics
├── adapters/         run any memory system through the same harness
└── experiment.py     the three-arm experiment + plots
```

**Retrieval** scores each memory by the generative-agents trio — recency, importance, relevance — as a weighted sum in `[0, 1]`, then multiplies by a **ranking strength** from the decay module.

**The decay module** (`DecayModule` ABC) is the pluggable axis. Three ship:
- `NoDecay` — keep everything at full strength (the hoarding baseline).
- `LastWriteWins` — a newer fact about the same *extracted* slot evicts the older one; no time decay, so noise is never forgotten.
- `EbbinghausDecay` — retention `R(t) = exp(-t / S)` with importance-weighted stability `S = τ·(1 + gain·importance)` (important memories consolidate); **contradiction supersession** collapses a superseded fact's ranking strength; and **retention-threshold pruning** drops decayed entries so memory stays bounded. Ranking-strength (supersession) is kept separate from the time-decay curve (pruning) on purpose: multiplying the score by time-decay too would let a fresh irrelevant memory outrank an old but still-correct one.

**A note on the workload's phrasing.** Canonical updates are phrased in the same frame as the query (*"Alice's home city is now denver"* vs *"What is Alice's home city?"*), so stale and current versions of a slot are lexically near-identical and retrieval leans on recency + supersession to separate them — isolating the forgetting mechanism from a phrasing confound. The *hard* updates deliberately break that frame; that is the whole source of the contradiction floor.

## Comparing against real libraries (mem0 / Letta / Zep)

The `MemoryAdapter` interface (`adapters/base.py`) is the seam: any system that ingests an observation and answers a query with ranked memories can be scored by the same harness. `adapters/reference.py` wraps this repo's core; `adapters/mem0_stub.py` is a **documented, non-wired stub** showing exactly where a real mem0 integration plugs in (including the metadata round-trip needed to score contradictions). The MVP takes no hard dependency on those packages.

## Status / what's next

- ✅ Reference core with three memory policies, imperfect slot extraction, seeded workload with ground truth, forgetting-axis metrics, one-command reproducible experiment with plots, 44 tests.
- 🔜 **Live incumbent adapters** — run mem0 / Letta / Zep through the harness for a real head-to-head (interface + stub exist; backends not wired).
- 🔜 **Semantic embeddings & LLM extraction** behind the existing interfaces — better extraction would *lower* the contradiction floor and *raise* recall on hard updates; the interesting question is whether decay's memory-bounding win survives, and it should.

### Honest limitations

- **Lexical, not semantic.** Relevance is bag-of-words overlap, so paraphrased current facts are hard to retrieve for *every* arm — which is why recall tops out around 0.67 here. That caps all arms equally (a fair comparison), but the absolute recall numbers would rise with real embeddings.
- **One extractor, false-negatives only.** The keyword extractor models missed contradictions; it doesn't yet model *false* supersession (wrongly merging two different slots), which would let forgetting hurt recall in a second way. That's a known gap, not modeled.
- **Synthetic workload.** The finding should be re-validated on real multi-session traffic before claiming it transfers; the value here is a clean, reproducible testbed and a decay module with a measured, honest tradeoff — not a production benchmark result.

## License

MIT — see [LICENSE](LICENSE).
