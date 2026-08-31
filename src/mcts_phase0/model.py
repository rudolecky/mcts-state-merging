"""Model harness: load model, generate CoT traces, extract hidden states at
step boundaries, and score rollout continuations against a verifier.

Key design decisions (see the approved plan for the reasoning behind each):
- Hidden states are extracted via a single teacher-forced forward pass over
  the *entire* already-generated token sequence, not via generate()'s
  per-step incremental hidden-state tuples. Causal masking guarantees the
  hidden state at position i depends only on tokens [0, i], so this is
  correct and much simpler than stitching generate()'s incremental output.
- Step-boundary token indices are found by decoding *prefixes* of the known
  token-ID sequence (pure ID->text), never by re-encoding text back into
  token IDs (BPE decode->encode round trips are not guaranteed injective).
- Rollout continuations start from token IDs sliced directly out of the
  original generated sequence -- never decoded-and-re-tokenized text.
- N rollouts per snapshot are generated in one batched `num_return_sequences=N`
  call: this avoids an RNG-reseeding bug that would silently collapse all N
  samples to identical outputs, and turns N sequential decode loops into one
  batched loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .datasets.common import ANSWER_LINE_RE, STEP_LINE_RE, split_steps

VerifierFn = Callable[[object, list[str], str | None], tuple[bool, dict]]


@dataclass
class LoadedModel:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str
    num_hidden_layers: int


def load_model(model_name: str, device: str = "mps", dtype: torch.dtype = torch.float32) -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_hidden_layers=model.config.num_hidden_layers,
    )


def resolve_layers(num_hidden_layers: int) -> dict[str, int]:
    """Map {mid, 3/4, final} to indices into the `hidden_states` tuple, which
    has length num_hidden_layers + 1 (index 0 = embeddings, index
    num_hidden_layers = final layer output).
    """
    return {
        "mid": num_hidden_layers // 2,
        "3/4": round(num_hidden_layers * 3 / 4),
        "final": num_hidden_layers,
    }


def verify_final_hidden_state_is_post_norm(lm: LoadedModel, sample_text: str = "The answer is") -> bool:
    """Empirically check whether hidden_states[-1] (from the CausalLM
    wrapper) is pre- or post- the model's final norm, for whichever
    checkpoint is actually loaded -- don't assume the Llama-family
    convention holds without checking.

    Ground truth reference: the *base* model's `last_hidden_state` output
    (from calling the inner transformer directly, bypassing the LM head) is,
    by well-established HF convention, always post-final-norm -- it's
    exactly what gets fed to the LM head to produce logits. Comparing
    hidden_states[-1] against it is a well-founded check; comparing it
    against a manually-renormed hidden_states[-2] is NOT, since
    hidden_states[-2] is the *input to* the last decoder layer, not that
    layer's output -- an earlier, wrong version of this function made
    exactly that mistake.
    """
    inputs = lm.tokenizer(sample_text, return_tensors="pt").to(lm.device)
    base_model = lm.model.model  # inner transformer, no LM head
    with torch.no_grad():
        base_out = base_model(**inputs, output_hidden_states=True)
    return torch.allclose(base_out.hidden_states[-1], base_out.last_hidden_state, atol=1e-5, rtol=1e-4)


def build_chat_prompt_ids(lm: LoadedModel, user_content: str) -> torch.Tensor:
    """Return a 1D tensor of prompt token IDs via the model's chat template."""
    messages = [
        {
            "role": "system",
            "content": (
                "You solve reasoning puzzles by chaining explicit steps in the "
                "exact format requested, then output a single final Answer line."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    encoded = lm.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    # BatchEncoding indexed by int returns the underlying tokenizers.Encoding,
    # not a tensor slice -- pull the actual input_ids tensor out explicitly.
    return encoded["input_ids"][0]


def generate_traces(
    lm: LoadedModel,
    prompt_ids: torch.Tensor,
    num_traces: int,
    max_new_tokens: int,
    temperature: float,
) -> list[torch.Tensor]:
    """Generate num_traces sampled continuations in one batched call. Returns
    a list of 1D tensors, each the full (prompt + generated) sequence with
    trailing pad/EOS tokens trimmed off.
    """
    inputs = prompt_ids.unsqueeze(0).to(lm.device)
    attention_mask = torch.ones_like(inputs)
    with torch.no_grad():
        out = lm.model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            num_return_sequences=num_traces,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=lm.tokenizer.eos_token_id,
        )
    prompt_len = prompt_ids.shape[0]
    traces = []
    for i in range(num_traces):
        traces.append(_trim_trailing_pad(out[i], prompt_len, lm.tokenizer.eos_token_id))
    return traces


def _trim_trailing_pad(full_ids: torch.Tensor, prompt_len: int, eos_id: int) -> torch.Tensor:
    gen_part = full_ids[prompt_len:].tolist()
    if eos_id in gen_part:
        cut = gen_part.index(eos_id)
        return full_ids[: prompt_len + cut]
    return full_ids


def hidden_states_for_sequence(lm: LoadedModel, full_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """One teacher-forced forward pass; returns the full hidden_states tuple
    (num_hidden_layers + 1 tensors, each [1, seq_len, hidden_dim])."""
    with torch.no_grad():
        out = lm.model(input_ids=full_ids.unsqueeze(0).to(lm.device), output_hidden_states=True)
    return out.hidden_states


def next_token_entropy_from_hidden(lm: LoadedModel, final_hidden_vec) -> float:
    """Next-token entropy at an already-extracted "final"-layer hidden
    vector, via the model's own output head -- no forward pass needed,
    since the saved final-layer vector already IS the post-norm hidden
    state the head consumes (see verify_final_hidden_state_is_post_norm).
    Used for the entropy-stratified false-merge check, not the main search
    loop.
    """
    head = lm.model.get_output_embeddings()
    with torch.no_grad():
        h = torch.as_tensor(final_hidden_vec, dtype=head.weight.dtype, device=head.weight.device)
        logits = head(h)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum()
    return float(entropy.item())


def _scan_line_boundaries(tokenizer, prompt_len: int, full_ids: torch.Tensor, line_matches) -> list[int]:
    """Shared decode-only scan: for each line in the generated portion whose
    stripped text satisfies `line_matches(line) -> bool`, find the token
    index (into full_ids) at which that line's content is fully generated.
    Never re-encodes text back into token IDs -- see module docstring.
    """
    gen_ids = full_ids[prompt_len:].tolist()
    if not gen_ids:
        return []
    full_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    lines = full_text.split("\n")

    target_offsets = []
    matched = []
    cursor = 0
    for line in lines:
        cursor += len(line)
        target_offsets.append(cursor)
        matched.append(bool(line_matches(line.strip())))
        cursor += 1  # the newline separator

    boundaries = []
    target_idx = 0
    for j in range(1, len(gen_ids) + 1):
        if target_idx >= len(target_offsets):
            break
        decoded_len = len(tokenizer.decode(gen_ids[:j], skip_special_tokens=True))
        if decoded_len >= target_offsets[target_idx]:
            if matched[target_idx]:
                boundaries.append(prompt_len + j - 1)
            target_idx += 1
    return boundaries


def find_step_boundaries(tokenizer, prompt_len: int, full_ids: torch.Tensor) -> list[int]:
    """Return, for each "Step N:" line in the generated portion, the token
    index (into full_ids) at which that line's content is fully generated.
    """
    return _scan_line_boundaries(tokenizer, prompt_len, full_ids, lambda line: STEP_LINE_RE.match(line))


def find_answer_boundary(tokenizer, prompt_len: int, full_ids: torch.Tensor) -> int | None:
    """Return the token index where the first "Answer: ..." line in the
    generated portion completes, or None if there isn't one.
    """
    boundaries = _scan_line_boundaries(tokenizer, prompt_len, full_ids, lambda line: ANSWER_LINE_RE.match(line))
    return boundaries[0] if boundaries else None


def find_position_based_boundaries(prompt_len: int, full_ids: torch.Tensor, num_snapshots: int) -> list[int]:
    """Evenly-spaced token indices across the generated portion. Model-
    agnostic fallback for when zero "Step N:" boundaries are found -- e.g.
    R1-distill-style reasoning models, which think in free-form prose inside
    a <think> block rather than emitting discrete labeled steps at all
    (confirmed empirically: DeepSeek-R1-Distill-Qwen-1.5B produced zero
    "Step N:" lines on ProsQA and inconsistent ones on Countdown). Depth here
    means "relative position through the generation," not "reasoning step
    number" -- a real change in what step_idx represents for this data,
    disclosed as `boundary_kind="position"` on the resulting records.
    """
    gen_len = full_ids.shape[0] - prompt_len
    if gen_len < num_snapshots:
        return []
    fractions = [(i + 1) / (num_snapshots + 1) for i in range(num_snapshots)]
    return [prompt_len + int(round(f * (gen_len - 1))) for f in fractions]


def score_rollouts(
    lm: LoadedModel,
    instance: object,
    verifier_fn: VerifierFn,
    prefix_ids: torch.Tensor,
    prompt_len: int,
    num_rollouts: int,
    max_new_tokens: int,
    temperature: float,
    split_fn=split_steps,
) -> float:
    """Roll out num_rollouts continuations from prefix_ids (sliced directly
    from an original generated sequence -- never re-tokenized text) and
    return the fraction that verify correct against the ground truth.

    `prompt_len` is the length of the original prompt, and it matters: the
    verifier must see the WHOLE reasoning trace (the already-generated steps
    inside the prefix, plus the new continuation), not just the continuation.
    Decoding from prefix_len instead of prompt_len silently hands the
    verifier a chain that starts mid-way, which fails well-formedness
    checks and drives every V-hat to 0.

    `split_fn` defaults to the shared `Step N:`/`Answer:` convention every
    dataset but gsm8k_native uses -- an opt-in override for datasets whose
    answer lives in unstructured free text instead (see gsm8k_native.py).
    """
    inputs = prefix_ids.unsqueeze(0).to(lm.device)
    attention_mask = torch.ones_like(inputs)
    with torch.no_grad():
        out = lm.model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            num_return_sequences=num_rollouts,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=lm.tokenizer.eos_token_id,
        )
    correct = 0
    for i in range(num_rollouts):
        gen_part = out[i][prompt_len:].tolist()
        eos_id = lm.tokenizer.eos_token_id
        if eos_id in gen_part:
            gen_part = gen_part[: gen_part.index(eos_id)]
        text = lm.tokenizer.decode(gen_part, skip_special_tokens=True)
        step_bodies, answer_body = split_fn(text)
        ok, _info = verifier_fn(instance, step_bodies, answer_body)
        if ok:
            correct += 1
    return correct / num_rollouts
