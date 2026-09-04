"""A small PyTorch slot extractor -- real weights, real forward pass.

The keyword extractor was the honest crux of the original benchmark: it links
canonical ``(entity, attribute)`` phrasing and misses the paraphrases that
never restate the attribute. This module *replaces* that write-path extractor
with a trained encoder, without collapsing the gap.

Training data is labelled by :class:`KeywordSlotExtractor` (the teacher):
canonical facts get a slot; hard templates, value-only chatter, and noise get
``None``. The network is therefore a learned approximation of the same failure
mode -- not a model taught to decode ``"Alice relocated to Denver"``. Ground
truth the memory never sees is still how metrics are scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedder import tokenize
from .extractor import KeywordSlotExtractor, SlotExtractor

PAD_ID = 0
UNK_ID = 1
MAX_LEN = 24
NONE_LABEL = 0


def _workload_vocab():
    # Imported lazily so extractor.py stays torch-free.
    from ..workload.synthetic import ATTRIBUTES, ENTITIES

    return ENTITIES, ATTRIBUTES


def keyword_teacher() -> KeywordSlotExtractor:
    from ..workload.synthetic import ATTRIBUTES, ENTITIES

    return KeywordSlotExtractor(
        entities=list(ENTITIES),
        attribute_phrases={spec.phrase: spec.key for spec in ATTRIBUTES.values()},
    )


def build_vocab() -> dict[str, int]:
    """Closed, sorted vocabulary over the synthetic workload's surface words."""
    from ..workload.synthetic import ATTRIBUTES, ENTITIES, _NOISE_PREDICATES, _NOISE_SUBJECTS

    tokens: set[str] = set()
    for entity in ENTITIES:
        tokens.update(tokenize(entity))
    for spec in ATTRIBUTES.values():
        tokens.update(tokenize(spec.phrase))
        tokens.update(tokenize(spec.key.replace("_", " ")))
        for value in spec.values:
            tokens.update(tokenize(value))
        for tmpl in spec.hard_templates:
            tokens.update(tokenize(tmpl.format(E="alice", v="boston")))
    for subject in _NOISE_SUBJECTS:
        tokens.update(tokenize(subject))
    for pred in _NOISE_PREDICATES:
        tokens.update(tokenize(pred))
    tokens.update(
        tokenize(
            "is now a to the in of with as was got upgraded switched moved leads "
            "settled relocated promoted obsessed eating raise earns working started "
            "mentioned coworker stranger weather team news podcast printer someone "
            "discussed note heard"
        )
    )
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for index, token in enumerate(sorted(tokens), start=2):
        vocab[token] = index
    return vocab


def encode_tokens(text: str, vocab: dict[str, int], max_len: int = MAX_LEN) -> list[int]:
    ids = [vocab.get(tok, UNK_ID) for tok in tokenize(text)[:max_len]]
    if len(ids) < max_len:
        ids.extend([PAD_ID] * (max_len - len(ids)))
    return ids


class SlotTagger(nn.Module):
    """Mean-pooled token embeddings plus a two-head MLP (entity, attribute)."""

    def __init__(
        self,
        vocab_size: int,
        n_entities: int,
        n_attributes: int,
        emb_dim: int = 32,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # +1 class on each head is NONE (index 0).
        self.entity_head = nn.Linear(hidden, n_entities + 1)
        self.attr_head = nn.Linear(hidden, n_attributes + 1)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # token_ids: [B, T]
        mask = (token_ids != PAD_ID).unsqueeze(-1).float()
        embedded = self.emb(token_ids)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        hidden = self.encoder(pooled)
        return self.entity_head(hidden), self.attr_head(hidden)


def _enumerate_training_texts() -> list[str]:
    from ..workload.synthetic import ATTRIBUTES, ENTITIES, _NOISE_PREDICATES, _NOISE_SUBJECTS

    texts: list[str] = []
    for entity in ENTITIES:
        title = entity.title()
        for spec in ATTRIBUTES.values():
            for value in spec.values:
                texts.append(f"{title}'s {spec.phrase} is {value}.")
                texts.append(f"{title}'s {spec.phrase} is now {value}.")
                texts.append(f"Note: {title}'s {spec.phrase} is {value}.")
                for tmpl in spec.hard_templates:
                    texts.append(tmpl.format(E=title, v=value))
                # Value mentioned with no attribute -- blocks value→slot leakage.
                texts.append(f"Someone mentioned {value}.")
                texts.append(f"A podcast discussed {value}.")
        texts.append(f"{title} said hello.")
        texts.append(f"{title} walked outside.")
    # Attribute phrase, no known entity -- teacher labels None; stops the
    # entity head from inventing a person just because "home city" fired.
    for spec in ATTRIBUTES.values():
        for value in spec.values:
            texts.append(f"the {spec.phrase} is {value}.")
            texts.append(f"someone's {spec.phrase} is {value}.")
    for subject in _NOISE_SUBJECTS:
        for pred in _NOISE_PREDICATES:
            texts.append(f"{subject.title()} {pred}.")
    return texts


def _label_texts(
    texts: list[str],
    teacher: KeywordSlotExtractor,
    entities: list[str],
    attributes: list[str],
) -> tuple[list[int], list[int]]:
    entity_index = {name: i + 1 for i, name in enumerate(entities)}
    attr_index = {name: i + 1 for i, name in enumerate(attributes)}
    y_ent: list[int] = []
    y_attr: list[int] = []
    for text in texts:
        extracted = teacher.extract(text)
        if extracted is None:
            y_ent.append(NONE_LABEL)
            y_attr.append(NONE_LABEL)
        else:
            entity, attribute = extracted
            y_ent.append(entity_index[entity])
            y_attr.append(attr_index[attribute])
    return y_ent, y_attr


@dataclass
class ExtractorTrainResult:
    steps: int
    final_loss: float


def train_slot_tagger(
    seed: int = 0,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 2e-2,
) -> tuple[SlotTagger, dict[str, int], list[str], list[str], ExtractorTrainResult]:
    """Train on teacher-labelled canonical / hard / noise text. Seeded, CPU-only."""
    entities, attributes_map = _workload_vocab()
    attributes = list(attributes_map.keys())
    teacher = keyword_teacher()
    vocab = build_vocab()
    texts = _enumerate_training_texts()
    y_ent, y_attr = _label_texts(texts, teacher, list(entities), attributes)

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = SlotTagger(len(vocab), len(entities), len(attributes))
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    encoded = torch.tensor([encode_tokens(t, vocab) for t in texts], dtype=torch.long)
    y_ent_t = torch.tensor(y_ent, dtype=torch.long)
    y_attr_t = torch.tensor(y_attr, dtype=torch.long)
    n = encoded.size(0)
    last_loss = 0.0
    steps = 0
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            ent_logits, attr_logits = model(encoded[idx])
            loss = F.cross_entropy(ent_logits, y_ent_t[idx]) + F.cross_entropy(
                attr_logits, y_attr_t[idx]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
            steps += 1

    model.eval()
    return model, vocab, list(entities), attributes, ExtractorTrainResult(steps, last_loss)


class LearnedSlotExtractor(SlotExtractor):
    """PyTorch :class:`SlotExtractor` used as the memory core's default write path."""

    def __init__(
        self,
        model: SlotTagger,
        vocab: dict[str, int],
        entities: list[str],
        attributes: list[str],
        threshold: float = 0.55,
    ) -> None:
        self.model = model
        self.model.eval()
        self.vocab = vocab
        self.entities = entities
        self.attributes = attributes
        self.threshold = threshold

    @classmethod
    def trained(cls, seed: int = 0, threshold: float = 0.55) -> LearnedSlotExtractor:
        model, vocab, entities, attributes, _ = train_slot_tagger(seed=seed)
        return cls(model, vocab, entities, attributes, threshold=threshold)

    def extract(self, text: str) -> tuple[str, str] | None:
        ids = torch.tensor([encode_tokens(text, self.vocab)], dtype=torch.long)
        with torch.no_grad():
            ent_logits, attr_logits = self.model(ids)
            ent_prob = F.softmax(ent_logits, dim=-1)[0]
            attr_prob = F.softmax(attr_logits, dim=-1)[0]
            ent_idx = int(ent_prob.argmax().item())
            attr_idx = int(attr_prob.argmax().item())
            if (
                ent_idx == NONE_LABEL
                or attr_idx == NONE_LABEL
                or float(ent_prob[ent_idx]) < self.threshold
                or float(attr_prob[attr_idx]) < self.threshold
            ):
                return None
            entity = self.entities[ent_idx - 1]
            attribute = self.attributes[attr_idx - 1]
            # Constrained decode: never emit an entity whose name is not in the
            # text. The network still scores the heads; this only blocks
            # hallucinations like tagging "the home city is boston" as Grace.
            if entity.lower() not in tokenize(text):
                return None
            return entity, attribute


_DEFAULT: LearnedSlotExtractor | None = None


def default_learned_extractor() -> LearnedSlotExtractor:
    """Cached, seed-0 extractor so ``make bench`` and tests share one trained net."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LearnedSlotExtractor.trained(seed=0)
    return _DEFAULT
