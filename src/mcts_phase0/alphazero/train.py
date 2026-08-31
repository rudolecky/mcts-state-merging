"""Replay buffer + training step. No arena/gatekeeping step: always trains
from and keeps the latest network -- the simplified "AlphaZero" convention
(not the original AlphaGo Zero's "only promote the new network if it beats
the old one"), a stated simplification per this plan's own design section,
not a silent omission.
"""

from __future__ import annotations

import random
from collections import deque

import torch
import torch.nn.functional as F

from ..datasets.connect_four_engine import legal_moves
from .network import WIDTH, encode_board
from .puct import HEIGHT
from .selfplay import SelfPlayExample


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[SelfPlayExample] = deque(maxlen=capacity)

    def add_game(self, examples: list[SelfPlayExample]) -> None:
        self.buffer.extend(examples)

    def sample(self, batch_size: int, rng: random.Random) -> list[SelfPlayExample]:
        return rng.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


def train_step(net, optimizer, batch: list[SelfPlayExample], device: str) -> dict[str, float]:
    net.train()
    boards = torch.stack([encode_board(ex.board, ex.to_move) for ex in batch]).to(device)

    policy_targets = torch.zeros(len(batch), WIDTH)
    # a large finite negative, not literal -inf: the target is legitimately 0 for every
    # masked (illegal) column, and 0 * (-inf) is NaN under IEEE float semantics, not 0 --
    # masking with -inf here would make every batch containing any illegal move NaN out.
    mask = torch.full((len(batch), WIDTH), -1e9)
    for i, ex in enumerate(batch):
        for col, p in ex.policy.items():
            policy_targets[i, col] = p
        mask[i, legal_moves(ex.board, HEIGHT)] = 0.0
    policy_targets = policy_targets.to(device)
    mask = mask.to(device)
    value_targets = torch.tensor([ex.outcome for ex in batch], dtype=torch.float32, device=device)

    policy_logits, values = net(boards)
    # masked cross-entropy against the MCTS visit-count target -- masking here matches
    # legal_policy's masking at inference time, so train and eval stay consistent.
    log_probs = F.log_softmax(policy_logits + mask, dim=-1)
    policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(values, value_targets)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item()}
