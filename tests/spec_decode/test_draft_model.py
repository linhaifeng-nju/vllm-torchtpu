# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.spec_decode.draft_model import DraftModelProposer
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer


def _config(draft_tp=1, target_tp=8):
    spec = SimpleNamespace(
        draft_tensor_parallel_size=draft_tp,
        method="draft_model",
        draft_model_config=SimpleNamespace(is_hybrid=False, get_vocab_size=lambda: 3),
        uses_draft_model=lambda: True,
        use_heterogeneous_vocab=False,
        draft_sample_method="probabilistic",
        num_speculative_tokens=2,
    )
    return SimpleNamespace(
        speculative_config=spec,
        parallel_config=SimpleNamespace(tensor_parallel_size=target_tp),
    )


@pytest.mark.parametrize("feature", ["use_heterogeneous_vocab", "is_hybrid"])
def test_unsupported_draft_semantics_fail_explicitly(feature):
    config = _config()
    if feature == "use_heterogeneous_vocab":
        config.speculative_config.use_heterogeneous_vocab = True
    else:
        config.speculative_config.draft_model_config.is_hybrid = True
    with pytest.raises(NotImplementedError):
        DraftModelProposer(SimpleNamespace(), config)


@pytest.mark.parametrize("filtered", [False, True])
def test_probabilistic_draft_returns_its_actual_sampling_distribution(filtered):
    logits = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    temperature = torch.tensor([[0.0], [0.7]])
    top_k = torch.tensor([[0], [1 if filtered else 0]])
    top_p = torch.ones(2, 1)
    uniform = torch.tensor([[0.2, 0.9, 0.1], [0.9, 0.2, 0.1]])
    tokens, probs = DraftModelProposer._sample_probabilistic.__wrapped__(
        logits, temperature, top_k, top_p, uniform, filtered
    )
    assert tokens[0] == 2
    torch.testing.assert_close(probs[0], torch.tensor([0.0, 0.0, 1.0]))
    if filtered:
        assert tokens[1] == 2
        torch.testing.assert_close(probs[1], torch.tensor([0.0, 0.0, 1.0]))
    else:
        torch.testing.assert_close(probs[1], torch.softmax(logits[1] / 0.7, dim=-1))
        noise = -torch.log(-torch.log(uniform[1]))
        assert tokens[1] == (logits[1] / 0.7 + noise).argmax()


def test_draft_probabilities_follow_request_reordering_and_verify_slots():
    proposer = DraftModelProposer(SimpleNamespace(device="cpu"), _config())
    proposer._draft_probs = {
        "a": [torch.tensor([0.2, 0.3, 0.5]), torch.tensor([0.5, 0.2, 0.3])],
        "b": [torch.tensor([0.1, 0.8, 0.1]), torch.tensor([0.6, 0.1, 0.3])],
    }
    probs = proposer.get_draft_probs(
        ["b", "new", "a"],
        torch.tensor([0, 0, 2, 3]),
        torch.tensor([0, 1, 1, 0]),
        np.array([2, 0, 2]),
    )
    torch.testing.assert_close(
        probs,
        torch.tensor([[0.1, 0.8, 0.1], [0.6, 0.1, 0.3], [0.5, 0.2, 0.3], [0, 0, 0]]),
    )


def test_missing_probabilities_for_a_real_proposal_fail():
    proposer = DraftModelProposer(SimpleNamespace(device="cpu"), _config())
    with pytest.raises(AssertionError, match="Missing draft probabilities"):
        proposer.get_draft_probs(
            ["missing"], torch.tensor([0]), torch.tensor([0]), np.array([1])
        )


def test_probabilistic_drafts_sample_the_declared_distribution():
    draws = 20000
    logits = torch.tensor([[0.0, 0.5, 1.0]]).expand(draws, -1)
    generator = torch.Generator().manual_seed(17)
    tokens, probs = DraftModelProposer._sample_probabilistic.__wrapped__(
        logits,
        torch.ones(draws, 1),
        torch.zeros(draws, 1, dtype=torch.int32),
        torch.ones(draws, 1),
        torch.rand(draws, 3, generator=generator),
        False,
    )
    empirical = torch.bincount(tokens.long(), minlength=3).float() / draws
    torch.testing.assert_close(empirical, probs[0], atol=0.015, rtol=0)


@pytest.mark.parametrize("rejected", [0, 1, 2])
@pytest.mark.parametrize("method", ["greedy", "probabilistic"])
def test_standalone_seed_follows_accepted_prefix_after_context_replay(
    monkeypatch, rejected, method
):
    runner = SimpleNamespace(
        device="cpu",
        input_batch=SimpleNamespace(req_ids=["active"]),
        requests={"active": None, "waiting": None},
    )
    config = _config()
    config.speculative_config.draft_sample_method = method
    proposer = DraftModelProposer(runner, config)
    previous = [torch.tensor([0.2, 0.3, 0.5])]
    proposer._draft_probs = {rid: previous for rid in ("active", "waiting", "finished")}
    ids = torch.tensor([5, 6, 7, 8])
    positions = torch.tensor([10, 11, 12, 13])
    index = torch.tensor([3 - rejected])
    seeded = ids.index_put((index,), torch.tensor([99]))
    chunk = SimpleNamespace(
        input_ids=ids,
        position_ids=positions,
        hidden_states=torch.ones(4, 2),
        start_index=0,
        num_reqs=1,
    )
    replayed = []
    monkeypatch.setattr(
        Eagle3Proposer,
        "_prepare_draft_inputs",
        lambda *_args: (seeded, positions, index, np.array([rejected])),
    )
    monkeypatch.setattr(
        Eagle3Proposer,
        "_forward_draft",
        lambda self, chunk, input_ids, positions, *_args, **_kwargs: replayed.append(
            (input_ids, positions)
        ),
    )
    seed, seed_pos, indices, _ = proposer._prepare_draft_inputs(
        chunk, [[99]], [], None, SimpleNamespace()
    )
    assert torch.equal(replayed[0][0], ids)
    assert torch.equal(replayed[0][1], positions)
    assert seed.tolist() == [99]
    assert seed_pos.tolist() == [14 - rejected]
    assert indices.tolist() == [0]
    if method == "probabilistic":
        assert set(proposer._draft_probs) == {"active", "waiting"}
        assert proposer._draft_probs["active"] == []
        assert proposer._draft_probs["waiting"] is previous


@pytest.mark.parametrize("uses_mrope", [False, True])
def test_idle_dp_rank_replays_context_then_seed_and_draft_loop(monkeypatch, uses_mrope):
    import vllm_torchtpu.spec_decode.eagle3 as eagle3

    monkeypatch.setattr(eagle3, "synchronize_tensors", lambda *_args, **_kwargs: None)
    runner = SimpleNamespace(
        _dp_target_bucket=64,
        _dp_step_max_reqs=8,
        num_reqs_max_model_len=64,
        num_tokens_paddings=[16, 32, 64],
        device="cpu",
        _hidden_states_dtype=torch.float32,
        uses_mrope=uses_mrope,
        _dp_lockstep_enabled=lambda: True,
    )
    proposer = DraftModelProposer(runner, _config(draft_tp=8))
    proposer.draft_model = SimpleNamespace(config=SimpleNamespace(hidden_size=8))
    proposer._draft_propose_token = lambda hidden: hidden[:, 0].to(torch.int32)
    forwards = []

    def forward(
        self,
        chunk,
        input_ids,
        positions,
        target_hidden_states,
        step_idx,
        seq_lens_delta,
        num_rejected_np,
        num_tokens_padded=None,
        num_rejected_dev=None,
        loop_query_start_loc=None,
        loop_request_distribution=None,
    ):
        if step_idx > 0:
            assert (
                chunk.num_reqs <= input_ids.shape[0] or loop_query_start_loc is not None
            )
        forwards.append((input_ids.shape[0], positions.ndim, step_idx, seq_lens_delta))
        return torch.zeros((input_ids.shape[0], 8)), None

    monkeypatch.setattr(Eagle3Proposer, "_forward_draft", forward)
    proposer.run_dp_dummy_draft(2)
    assert forwards == [
        (64, 1, 0, 0),
        (16, 1, 1, 1),
        (64, 1, 0, 0),
        (16, 1, 1, 1),
        (16, 1, 2, 2),
        (16, 1, 2, 2),
    ]
