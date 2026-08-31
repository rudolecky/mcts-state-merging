"""A small value+policy network for Connect Four (the project's standard
5-wide, 4-tall board), in the AlphaZero style: two convolutional heads over
a shared trunk, trained end-to-end via self-play. This is the first
trained-from-scratch network in this project -- everywhere else, `torch`
was only ever used for inference against a frozen pretrained LLM.

Input encoding is the standard AlphaZero canonicalization: two binary
planes, "pieces belonging to whoever moves next" and "the opponent's
pieces" -- never "X's pieces"/"O's pieces" -- so the network never needs to
learn the same strategy twice under two different labels.

Value convention is the real AlphaZero one, not this project's usual fixed
hero-perspective: the value head outputs how good the position is for
`to_move`, the player about to act, in [0, 1]. This is a deliberate
departure from every other module in this project (which fixes the
perspective to one hero and never flips it) -- necessary here because
real PUCT backup flips the value's sign at every ply as it walks back up
the tree (see puct.py's `backup`), which only makes sense with a
current-player-relative value in the first place.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datasets.connect_four_engine import legal_moves

WIDTH = 5
HEIGHT = 4


def encode_board(board: tuple, to_move: str) -> torch.Tensor:
    """(2, HEIGHT, WIDTH) tensor: plane 0 is `to_move`'s own pieces, plane 1
    is the opponent's -- board-tuple row order (0 = bottom) used directly,
    consistent with connect_four_engine.py's own convention."""
    planes = torch.zeros(2, HEIGHT, WIDTH)
    for col, column in enumerate(board):
        for row, piece in enumerate(column):
            planes[0 if piece == to_move else 1, row, col] = 1.0
    return planes


def legal_policy(policy_logits: torch.Tensor, board: tuple) -> torch.Tensor:
    """Masks illegal columns to zero probability and renormalizes.
    `policy_logits` is a length-WIDTH 1D tensor (unbatched)."""
    moves = legal_moves(board, HEIGHT)
    mask = torch.full((WIDTH,), float("-inf"))
    mask[moves] = 0.0
    return F.softmax(policy_logits + mask, dim=-1)


class _ConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ConnectFourNet(nn.Module):
    """Sized for a 5x4 board -- a handful of small conv blocks is already
    generous here, not a scaled-down version of AlphaZero's own much larger
    ResNet (built for 19x19 Go)."""

    def __init__(self, channels: int = 48, num_blocks: int = 3):
        super().__init__()
        self.input_conv = nn.Conv2d(2, channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([_ConvBlock(channels) for _ in range(num_blocks)])

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * HEIGHT * WIDTH, WIDTH)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(HEIGHT * WIDTH, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, 2, HEIGHT, WIDTH). Returns (policy_logits, value),
        shapes (batch, WIDTH) and (batch,) -- value in [0, 1] via sigmoid."""
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.blocks:
            x = block(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_logits = self.policy_fc(p.flatten(1))

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        value = torch.sigmoid(self.value_fc2(v)).squeeze(-1)
        return policy_logits, value


@torch.no_grad()
def evaluate(net: ConnectFourNet, board: tuple, to_move: str, device: str = "cpu") -> tuple[torch.Tensor, float]:
    """One forward pass for one position. Returns (masked+renormalized
    legal-move policy over all WIDTH columns, scalar value in [0,1] from
    `to_move`'s own perspective)."""
    net.eval()
    x = encode_board(board, to_move).unsqueeze(0).to(device)
    policy_logits, value = net(x)
    policy = legal_policy(policy_logits.squeeze(0).cpu(), board)
    return policy, float(value.item())
