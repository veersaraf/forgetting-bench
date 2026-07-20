"""The headline experiment: decay ON vs decay OFF over one seeded workload.

Run with ``python -m forgetting_bench.experiment`` (or ``make bench``). It:

1. builds a deterministic long-horizon workload,
2. replays it through the reference core with ``NoDecay`` and ``EbbinghausDecay``,
3. sweeps decay stability (``tau``) to trace the forgetting-vs-recall tradeoff,
4. writes a results table (``results/summary.{json,md}``) and three plots.

Every number printed comes from the actual run -- nothing here is hard-coded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

from .adapters.reference import ReferenceAdapter  # noqa: E402
from .bench.harness import BenchResult, run  # noqa: E402
from .bench.metrics import QueryEval, contradiction_rate, recall_rate  # noqa: E402
from .memory import EbbinghausDecay, NoDecay  # noqa: E402
from .workload.synthetic import generate_workload  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@dataclass
class ABConfig:
    seed: int = 0
    n_turns: int = 3000
    k: int = 5
    tau: float = 150.0  # headline operating point (on the recall plateau)


def run_ab(cfg: ABConfig) -> dict[str, BenchResult]:
    workload = generate_workload(seed=cfg.seed, n_turns=cfg.n_turns)
    off = run(ReferenceAdapter(NoDecay(), name="decay-off (baseline)"), workload, k=cfg.k)
    on = run(
        ReferenceAdapter(EbbinghausDecay(tau=cfg.tau), name="decay-on (Ebbinghaus)"),
        workload,
        k=cfg.k,
    )
    return {"off": off, "on": on}


def sweep_tau(cfg: ABConfig, taus: list[float]) -> list[tuple[float, float, float, int]]:
    """Return (tau, recall_rate, contradiction_rate, final_size) for each tau."""
    workload = generate_workload(seed=cfg.seed, n_turns=cfg.n_turns)
    rows: list[tuple[float, float, float, int]] = []
    for tau in taus:
        res = run(ReferenceAdapter(EbbinghausDecay(tau=tau)), workload, k=cfg.k)
        rows.append((tau, res.recall_rate, res.contradiction_rate, res.final_size))
    return rows


# -- plotting --------------------------------------------------------------


def _bucketed_contradiction(evals: list[QueryEval], n_turns: int, bins: int = 20):
    width = max(1, n_turns // bins)
    xs, ys = [], []
    for b in range(bins):
        lo, hi = b * width, (b + 1) * width
        chunk = [e for e in evals if lo <= e.turn < hi]
        if chunk:
            xs.append((lo + hi) / 2)
            ys.append(contradiction_rate(chunk))
    return xs, ys


def plot_bloat(results: dict[str, BenchResult], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for res in results.values():
        turns = [t for t, _ in res.size_trace]
        sizes = [s for _, s in res.size_trace]
        ax.plot(turns, sizes, label=res.name, linewidth=2)
    ax.set_xlabel("turn")
    ax.set_ylabel("live memories held")
    ax.set_title("Memory bloat over a long horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_contradiction(results: dict[str, BenchResult], n_turns: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for res in results.values():
        xs, ys = _bucketed_contradiction(res.evals, n_turns)
        ax.plot(xs, ys, label=res.name, linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("turn")
    ax.set_ylabel("stale-fact contradiction rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Do retrievals surface contradicted, stale facts?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_tradeoff(
    rows: list[tuple[float, float, float, int]], baseline_size: int, path: Path
) -> None:
    """Forgetting-vs-recall frontier: recall against how much memory is kept."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sizes = [s for _, _, _, s in rows]
    recalls = [r for _, r, _, _ in rows]
    ax.plot(sizes, recalls, "-o", color="#1f6feb", linewidth=2)
    for tau, rec, _, size in rows:
        ax.annotate(f"τ={tau:g}", (size, rec), textcoords="offset points",
                    xytext=(6, -10), fontsize=8)
    ax.axvline(baseline_size, color="#888", linestyle="--", linewidth=1)
    ax.annotate("baseline\n(keep everything)", (baseline_size, min(recalls)),
                textcoords="offset points", xytext=(-95, 5), fontsize=8, color="#555")
    ax.set_xlabel("memory kept (live entries at end of run)")
    ax.set_ylabel("recall rate (current fact retrieved)")
    ax.set_title("Forgetting-vs-recall frontier (sweeping decay stability τ)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# -- reporting -------------------------------------------------------------


def _format_table(off: BenchResult, on: BenchResult) -> str:
    o, n = off.summary(), on.summary()
    rows = [
        ("Stale-fact contradiction rate", "contradiction_rate", "lower"),
        ("Stale context fraction", "stale_context_rate", "lower"),
        ("Recall rate (current fact found)", "recall_rate", "higher"),
        ("Answer accuracy (top-1 correct)", "answer_accuracy", "higher"),
        ("Precision (correct fact / retrieved)", "mean_precision", "higher"),
        ("Final memory size", "final_size", "lower"),
        ("Peak memory size", "peak_size", "lower"),
        ("Final token count", "final_tokens", "lower"),
    ]
    lines = [
        f"| Metric (n={int(o['n_queries'])} queries) | decay-off | decay-on | better |",
        "|---|---|---|---|",
    ]
    for label, key, better in rows:
        if key in ("final_size", "peak_size", "final_tokens"):
            ov, nv = f"{int(o[key])}", f"{int(n[key])}"
        else:
            ov, nv = f"{o[key]:.3f}", f"{n[key]:.3f}"
        lines.append(f"| {label} | {ov} | {nv} | {better} |")
    return "\n".join(lines)


def main() -> None:
    cfg = ABConfig()
    RESULTS_DIR.mkdir(exist_ok=True)

    results = run_ab(cfg)
    off, on = results["off"], results["on"]

    table = _format_table(off, on)
    print("\nForgetting-Bench A/B  (seed={}, {} turns, k={})\n".format(
        cfg.seed, cfg.n_turns, cfg.k))
    print(table)

    taus = [15, 30, 60, 120, 240, 500]
    sweep = sweep_tau(cfg, taus)

    plot_bloat(results, RESULTS_DIR / "bloat.png")
    plot_contradiction(results, cfg.n_turns, RESULTS_DIR / "contradiction.png")
    plot_tradeoff(sweep, off.final_size, RESULTS_DIR / "tradeoff.png")

    summary = {
        "config": vars(cfg),
        "decay_off": off.summary(),
        "decay_on": on.summary(),
        "tau_sweep": [
            {"tau": t, "recall_rate": r, "contradiction_rate": c, "final_size": s}
            for t, r, c, s in sweep
        ],
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (RESULTS_DIR / "summary.md").write_text(
        f"# Forgetting-Bench A/B results\n\n"
        f"seed={cfg.seed}, n_turns={cfg.n_turns}, k={cfg.k}\n\n{table}\n"
    )
    print(f"\nWrote plots + summary to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
