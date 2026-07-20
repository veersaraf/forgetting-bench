"""The headline experiment: three memory policies over one seeded workload.

Run with ``python -m forgetting_bench.experiment`` (or ``make bench``). It:

1. builds a deterministic long-horizon workload (40% of updates are paraphrased
   or implicit, so keyword slot-extraction genuinely misses some contradictions),
2. replays it through the reference core under three policies -- keep-everything,
   last-write-wins dedup (the mem0-style bar), and Ebbinghaus decay,
3. sweeps decay stability (``tau``) to trace the forgetting-vs-recall frontier,
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
from .bench.metrics import QueryEval, contradiction_rate  # noqa: E402
from .memory import EbbinghausDecay, LastWriteWins, NoDecay  # noqa: E402
from .workload.synthetic import generate_workload  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Arm labels, in report order.
KEEP_ALL = "keep-everything"
LWW = "last-write-wins"
DECAY = "ebbinghaus-decay"


@dataclass
class ABConfig:
    seed: int = 0
    n_turns: int = 3000
    k: int = 5
    tau: float = 150.0  # headline operating point (matches LWW recall, prunes noise)


def run_arms(cfg: ABConfig) -> dict[str, BenchResult]:
    workload = generate_workload(seed=cfg.seed, n_turns=cfg.n_turns)
    arms = {
        KEEP_ALL: NoDecay(),
        LWW: LastWriteWins(),
        DECAY: EbbinghausDecay(tau=cfg.tau),
    }
    return {
        name: run(ReferenceAdapter(decay, name=name), workload, k=cfg.k)
        for name, decay in arms.items()
    }


def sweep_tau(cfg: ABConfig, taus: list[float]) -> list[tuple[float, float, float, int]]:
    """Return (tau, recall_rate, contradiction_rate, final_size) for each tau."""
    workload = generate_workload(seed=cfg.seed, n_turns=cfg.n_turns)
    rows: list[tuple[float, float, float, int]] = []
    for tau in taus:
        res = run(ReferenceAdapter(EbbinghausDecay(tau=tau)), workload, k=cfg.k)
        rows.append((tau, res.recall_rate, res.contradiction_rate, res.final_size))
    return rows


# -- plotting --------------------------------------------------------------

_COLORS = {KEEP_ALL: "#b02418", LWW: "#d98c00", DECAY: "#1f6feb"}


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
    for name, res in results.items():
        turns = [t for t, _ in res.size_trace]
        sizes = [s for _, s in res.size_trace]
        ax.plot(turns, sizes, label=name, linewidth=2, color=_COLORS[name])
    ax.set_xlabel("turn")
    ax.set_ylabel("live memories held")
    ax.set_title("Memory growth over a long horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_contradiction(results: dict[str, BenchResult], n_turns: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, res in results.items():
        xs, ys = _bucketed_contradiction(res.evals, n_turns)
        ax.plot(xs, ys, label=name, linewidth=2, marker="o", markersize=3,
                color=_COLORS[name])
    ax.set_xlabel("turn")
    ax.set_ylabel("stale-fact contradiction rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Do retrievals surface contradicted, stale facts?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_frontier(
    rows: list[tuple[float, float, float, int]],
    results: dict[str, BenchResult],
    path: Path,
) -> None:
    """The money plot: recall vs contradiction, tracing decay's tunable frontier
    against the two fixed baselines."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    recalls = [r for _, r, _, _ in rows]
    contradictions = [c for _, _, c, _ in rows]
    ax.plot(recalls, contradictions, "-o", color=_COLORS[DECAY], linewidth=2,
            label="ebbinghaus-decay (sweep τ)", zorder=3)
    # Annotate only the moving part of the curve; high-τ points pile up.
    annotate = {rows[0][0], rows[1][0], rows[2][0], rows[-1][0]}
    for tau, rec, con, _ in rows:
        if tau in annotate:
            ax.annotate(f"τ={tau:g}", (rec, con), textcoords="offset points",
                        xytext=(6, -12), fontsize=8, color="#555")
    for name in (KEEP_ALL, LWW):
        res = results[name]
        ax.scatter([res.recall_rate], [res.contradiction_rate], s=90, zorder=4,
                   color=_COLORS[name], label=name, edgecolor="white")
    ax.set_xlabel("recall rate (current fact retrieved)  →  better")
    ax.set_ylabel("contradiction rate (stale fact retrieved)  →  worse")
    ax.set_title("Forgetting-vs-recall frontier")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# -- reporting -------------------------------------------------------------

_TABLE_ROWS = [
    ("Stale-fact contradiction rate", "contradiction_rate", "lower"),
    ("Stale context fraction", "stale_context_rate", "lower"),
    ("Recall rate (current fact found)", "recall_rate", "higher"),
    ("Answer accuracy (top-1 correct)", "answer_accuracy", "higher"),
    ("Precision (correct fact / retrieved)", "mean_precision", "higher"),
    ("Final memory size", "final_size", "lower"),
    ("Final token count", "final_tokens", "lower"),
]


def _format_table(results: dict[str, BenchResult]) -> str:
    names = list(results)
    n_q = int(next(iter(results.values())).summary()["n_queries"])
    header = f"| Metric (n={n_q} queries) | " + " | ".join(names) + " | want |"
    sep = "|" + "---|" * (len(names) + 2)
    lines = [header, sep]
    for label, key, better in _TABLE_ROWS:
        cells = []
        for name in names:
            v = results[name].summary()[key]
            cells.append(f"{int(v)}" if key in ("final_size", "final_tokens")
                         else f"{v:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {better} |")
    return "\n".join(lines)


def main() -> None:
    cfg = ABConfig()
    RESULTS_DIR.mkdir(exist_ok=True)

    results = run_arms(cfg)
    table = _format_table(results)
    print(f"\nForgetting-Bench  (seed={cfg.seed}, {cfg.n_turns} turns, k={cfg.k}, "
          f"tau={cfg.tau:g}, hard-update fraction=0.4)\n")
    print(table)

    taus = [30, 60, 120, 240, 500, 1000]
    sweep = sweep_tau(cfg, taus)

    plot_bloat(results, RESULTS_DIR / "bloat.png")
    plot_contradiction(results, cfg.n_turns, RESULTS_DIR / "contradiction.png")
    plot_frontier(sweep, results, RESULTS_DIR / "frontier.png")

    summary = {
        "config": vars(cfg),
        "arms": {name: res.summary() for name, res in results.items()},
        "tau_sweep": [
            {"tau": t, "recall_rate": r, "contradiction_rate": c, "final_size": s}
            for t, r, c, s in sweep
        ],
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (RESULTS_DIR / "summary.md").write_text(
        f"# Forgetting-Bench results\n\n"
        f"seed={cfg.seed}, n_turns={cfg.n_turns}, k={cfg.k}, tau={cfg.tau:g}\n\n"
        f"{table}\n"
    )
    print(f"\nWrote plots + summary to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
