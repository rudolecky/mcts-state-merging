"""The one correctness-critical empirical check on the model harness itself:
does a hidden state captured as part of a longer forward pass match a
hidden state computed from a forward pass over just the prefix ending at
that position? Causal masking guarantees this must hold; if it doesn't,
something is wrong with how hidden states are being read out, and nothing
built on top of it (the whole Phase 0 probe) can be trusted.

Marked slow: downloads and loads a real model. Run explicitly with:
    uv run pytest tests/test_model_consistency.py -m slow -v
"""

import pytest
import torch

from mcts_phase0.model import (
    build_chat_prompt_ids,
    generate_traces,
    hidden_states_for_sequence,
    load_model,
    verify_final_hidden_state_is_post_norm,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def lm():
    return load_model(MODEL_NAME, device="mps", dtype=torch.float32)


@pytest.mark.slow
def test_hidden_state_reproducible_from_prefix_only(lm):
    prompt_ids = build_chat_prompt_ids(lm, "What is 12 + 7? Answer with just the number.")
    traces = generate_traces(lm, prompt_ids, num_traces=1, max_new_tokens=20, temperature=0.8)
    full_ids = traces[0]
    assert full_ids.shape[0] > prompt_ids.shape[0], "nothing was generated"

    full_hidden = hidden_states_for_sequence(lm, full_ids)
    boundary = (prompt_ids.shape[0] + full_ids.shape[0]) // 2
    prefix_ids = full_ids[: boundary + 1]
    prefix_hidden = hidden_states_for_sequence(lm, prefix_ids)

    checked_layers = sorted({0, lm.num_hidden_layers // 2, lm.num_hidden_layers})
    for layer_idx in checked_layers:
        a = full_hidden[layer_idx][0, boundary, :]
        b = prefix_hidden[layer_idx][0, boundary, :]
        max_abs_diff = (a - b).abs().max().item()
        assert torch.allclose(a, b, atol=1e-3, rtol=1e-2), (
            f"layer {layer_idx}: max abs diff {max_abs_diff} -- hidden state at "
            f"position {boundary} differs between a full-sequence forward pass "
            f"and a prefix-only forward pass. This breaks the core assumption "
            f"the whole pipeline depends on."
        )


@pytest.mark.slow
def test_final_hidden_state_norm_convention(lm):
    is_post_norm = verify_final_hidden_state_is_post_norm(lm)
    print(f"\n[documented] hidden_states[-1] is post-final-norm: {is_post_norm}")
    assert isinstance(is_post_norm, bool)
