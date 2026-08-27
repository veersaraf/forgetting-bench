"""A learned forget policy -- ranking strength and prune from a tiny MLP.

Supersession still keys on the *extracted* slot, the same honest constraint as
:class:`EbbinghausDecay` and :class:`LastWriteWins`. What is learned is the
forget *decision*: given age, importance, whether the fact was contradicted,
and whether it was slotted at all, should this memory still rank, and should
it still occupy space?

The network is trained offline on synthetic feature vectors (not the benchmark
workload, and not ground-truth values). Labels come from a teacher that:

* crushes superseded facts (contradiction handling),
* forgets unslotted distractors faster than slotted current facts (the bound),
* keeps important slotted facts longer (consolidation).

At inference the policy sees only those four features -- never the metric's
ground truth.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decay import DecayModule
from .entry import MemoryEntry

# Feature layout: [elapsed/1000, importance, superseded, slotted]
N_FEATURES = 4


class ForgetNet(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(N_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.strength_head = nn.Linear(hidden, 1)
        self.retain_head = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        strength = torch.sigmoid(self.strength_head(hidden))
        retain_logit = self.retain_head(hidden)
        return strength.squeeze(-1), retain_logit.squeeze(-1)


def _teacher_targets(
    elapsed: torch.Tensor,
    importance: torch.Tensor,
    superseded: torch.Tensor,
    slotted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic teacher used only for training labels.

    Slotted facts follow an Ebbinghaus-like curve (tau=150). Unslotted noise
    is given a much shorter horizon so the learned policy actually bounds
    memory -- last-write-wins never forgets what it cannot slot.
    """
    turns = elapsed * 1000.0
    stability = 150.0 * (1.0 + 4.0 * importance)
    retention = torch.exp(-turns / stability)
    supersede_mask = superseded > 0.5
    slot_mask = slotted > 0.5
    retention = torch.where(supersede_mask, retention * 0.05, retention)
    # Unslotted distractors decay on a ~50-turn horizon so noise is actually
    # forgotten -- last-write-wins never drops what it cannot slot.
    noise_ret = torch.exp(-turns / 50.0)
    retention = torch.where(slot_mask, retention, noise_ret)
    prune = (retention < 0.02).float()
    strength = torch.where(
        supersede_mask,
        torch.full_like(retention, 0.05),
        torch.ones_like(retention),
    )
    retain = 1.0 - prune
    return strength, retain


def train_forget_net(
    seed: int = 0,
    steps: int = 400,
    batch_size: int = 64,
    lr: float = 1e-2,
) -> tuple[ForgetNet, float]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = ForgetNet()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last_loss = 0.0
    for _ in range(steps):
        elapsed = torch.rand(batch_size) * 3.0
        importance = torch.rand(batch_size)
        superseded = (torch.rand(batch_size) < 0.3).float()
        slotted = (torch.rand(batch_size) < 0.45).float()
        # A contradicted fact is always slotted -- that is how supersession fires.
        slotted = torch.clamp(slotted + superseded, max=1.0)
        features = torch.stack([elapsed, importance, superseded, slotted], dim=1)
        strength_t, retain_t = _teacher_targets(elapsed, importance, superseded, slotted)
        strength_p, retain_logit = model(features)
        loss = F.mse_loss(strength_p, strength_t) + F.binary_cross_entropy_with_logits(
            retain_logit, retain_t
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
    model.eval()
    return model, last_loss


class LearnedForget(DecayModule):
    """Learned ranking-strength + prune; slot supersession stays extraction-gated."""

    def __init__(
        self,
        model: ForgetNet | None = None,
        retain_threshold: float = 0.5,
    ) -> None:
        self.model = model if model is not None else train_forget_net()[0]
        self.model.eval()
        self.retain_threshold = retain_threshold

    @classmethod
    def trained(cls, seed: int = 0) -> LearnedForget:
        model, _ = train_forget_net(seed=seed)
        return cls(model=model)

    def _features(self, entry: MemoryEntry, now: int) -> torch.Tensor:
        elapsed = max(0, now - entry.turn) / 1000.0
        superseded = 1.0 if entry.superseded_by is not None else 0.0
        slotted = 1.0 if entry.slot is not None else 0.0
        return torch.tensor(
            [[elapsed, float(entry.importance), superseded, slotted]],
            dtype=torch.float32,
        )

    def _forward(self, entry: MemoryEntry, now: int) -> tuple[float, float]:
        with torch.no_grad():
            strength, retain_logit = self.model(self._features(entry, now))
            retain = float(torch.sigmoid(retain_logit).item())
            return float(strength.item()), retain

    def strength(self, entry: MemoryEntry, now: int) -> float:
        s, _ = self._forward(entry, now)
        return s

    def retention(self, entry: MemoryEntry, now: int) -> float:
        _, retain = self._forward(entry, now)
        return retain

    def on_add(self, new_entry: MemoryEntry, existing: Iterable[MemoryEntry]) -> None:
        slot = new_entry.slot
        if slot is None:
            return
        for entry in existing:
            if entry.id != new_entry.id and entry.slot == slot and entry.superseded_by is None:
                entry.superseded_by = new_entry.id

    def should_prune(self, entry: MemoryEntry, now: int) -> bool:
        return self.retention(entry, now) < self.retain_threshold
