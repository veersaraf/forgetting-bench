"""The headline experiment: memory policies over one seeded workload.

Run with ``python -m forgetting_bench.experiment`` (or ``make bench``). It:

1. builds a deterministic long-horizon workload (40% of updates are paraphrased
   or implicit, so slot-extraction genuinely misses some contradictions),
2. replays it through the reference core under keep-everything, last-write-wins
   dedup, Ebbinghaus decay, and a learned forget policy,
3. sweeps decay stability (``tau``) to trace the forgetting-vs-recall frontier,
4. writes a results table (``results/summary.{json,md}``) and three plots.

``--incumbents`` additionally constructs live mem0 / Letta adapters when the
SDK and credentials are present. If they are not runnable, the run prints the
probe reason and does **not** invent a score.

Every number printed comes from the actual run -- nothing here is hard-coded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

from .adapters.errors import AdapterUnavailable  # noqa: E402
from .adapters.probe import incumbent_probes  # noqa: E402
from .adapters.reference import ReferenceAdapter  # noqa: E402
from .bench.harness import BenchResult, run  # noqa: E402
from .bench.metrics import QueryEval, contradiction_rate  # noqa: E402
from .memory import EbbinghausDecay, LastWriteWins, LearnedForget, NoDecay  # noqa: E402
from .workload.synthetic import generate_workload  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Arm labels, in report order.
KEEP_ALL = "keep-everything"
LWW = "last-write-wins"
DECAY = "ebbinghaus-decay"
LEARNED = "learned-forget"


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
        LEARNED: LearnedForget.trained(seed=cfg.seed),
    }
    return {
        name: run(ReferenceAdapter(decay, name=name), workload, k=cfg.k)
        for name, decay in arms.items()
    }


def try_live_incumbents(cfg: ABConfig) -> tuple[dict[str, BenchResult], list[str]]:
    """Run live backends that probe as runnable. Skip (do not score) the rest."""
    from .adapters.letta import LettaAdapter
    from .adapters.mem0 import Mem0Adapter

    workload = generate_workload(seed=cfg.seed, n_turns=cfg.n_turns)
    results: dict[str, BenchResult] = {}
    skips: list[str] = []
    for name, factory in (("mem0", Mem0Adapter), ("letta", LettaAdapter)):
        try:
            adapter = factory()
        except AdapterUnavailable as exc:
            skips.append(str(exc))
            continue
        try:
            results[name] = run(adapter, workload, k=cfg.k)
        except Exception as exc:  # noqa: BLE001 -- live backend failed mid-run
            skips.append(f"{name} started but failed: {exc}")
    return results, skips


def sweep_tau(cfg: ABConfig, taus: list[float]) -> list[tuple[float, float, float, int]]:
    """Return (tau, recall_rate, contradiction_rate, final_size) for each tau."""
    workload = generate_workload(seed=cfg.seed, n_turns=cfg.n_turns)
    rows: list[tuple[float, float, float, int]] = []
    for tau in taus:
        res = run(ReferenceAdapter(EbbinghausDecay(tau=tau)), workload, k=cfg.k)
        rows.append((tau, res.recall_rate, res.contradiction_rate, res.final_size))
    return rows


# -- plotting --------------------------------------------------------------

_COLORS = {
    KEEP_ALL: "#b02418",
    LWW: "#d98c00",
    DECAY: "#1f6feb",
    LEARNED: "#6f42c1",
    "mem0": "#111111",
    "letta": "#2da44e",
}


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
        ax.plot(turns, sizes, label=name, linewidth=2, color=_COLORS.get(name, "#333"))
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
                color=_COLORS.get(name, "#333"))
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
    for name in (KEEP_ALL, LWW, LEARNED):
        if name not in results:
            continue
        res = results[name]
        ax.scatter([res.recall_rate], [res.contradiction_rate], s=90, zorder=4,
                   color=_COLORS.get(name, "#333"), label=name, edgecolor="white")
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forgetting-Bench headline experiment")
    parser.add_argument(
        "--incumbents",
        action="store_true",
        help="Also run live mem0 / Letta adapters when configured. "
        "Skips (does not invent results) when the SDK or credentials are missing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = ABConfig()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Live incumbent probes (not scored unless --incumbents and runnable):")
    for probe in incumbent_probes():
        print(f"  {probe.summary()}")

    results = run_arms(cfg)
    skipped: list[str] = []
    if args.incumbents:
        live, skipped = try_live_incumbents(cfg)
        results.update(live)

    table = _format_table(results)
    print(f"\nForgetting-Bench  (seed={cfg.seed}, {cfg.n_turns} turns, k={cfg.k}, "
          f"tau={cfg.tau:g}, hard-update fraction=0.4)\n")
    print(table)
    if skipped:
        print("\nIncumbents not scored:")
        for line in skipped:
            print(f"  - {line}")

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
        "incumbent_probes": [
            {"name": p.name, "runnable": p.runnable, "reason": p.reason}
            for p in incumbent_probes()
        ],
        "incumbents_skipped": skipped,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    skip_block = ""
    if skipped:
        skip_block = "\n\nIncumbents not scored:\n" + "\n".join(f"- {s}" for s in skipped)
    probes = "\n".join(f"- {p.summary()}" for p in incumbent_probes())
    (RESULTS_DIR / "summary.md").write_text(
        f"# Forgetting-Bench results\n\n"
        f"seed={cfg.seed}, n_turns={cfg.n_turns}, k={cfg.k}, tau={cfg.tau:g}\n\n"
        f"{table}\n\n"
        f"## Live incumbent probes\n\n{probes}{skip_block}\n"
    )
    print(f"\nWrote plots + summary to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
