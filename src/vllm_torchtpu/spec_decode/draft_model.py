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
"""Independent autoregressive drafting on TPU.

Follows vLLM v0.33.0's v1/spec_decode/draft_model.py: independent weights,
unshifted context tokens, and no target hidden-state input. Reuses the TPU
proposer's attention metadata and bucketed loop instead of its CUDA/Triton base.
"""

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.sampler import _SAMPLING_EPS

from vllm_torchtpu.layers.adapter.sample.top_k_top_p import apply_top_k_top_p
from vllm_torchtpu.spec_decode.eagle3 import (
    Eagle3Proposer,
    _DpDummyDraft,
    _static_shape_args,
)
from vllm_torchtpu.spec_decode.utils import DraftChunkInputs, fit_rows

if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class DraftModelProposer(Eagle3Proposer):
    """A standalone causal model with its own prompt and decode KV cache."""

    def __init__(self, runner: "TPUModelRunner", vllm_config: VllmConfig) -> None:
        super().__init__(runner, vllm_config)
        if self.speculative_config.use_heterogeneous_vocab:
            raise NotImplementedError(
                "Standalone TPU drafts require matching token vocabularies; "
                "use_heterogeneous_vocab is not implemented."
            )
        if self.speculative_config.draft_model_config.is_hybrid:
            raise NotImplementedError(
                "Standalone TPU draft models require attention KV; recurrent "
                "draft-state rollback is not implemented."
            )
        self._sampling_chunk: DraftChunkInputs | None = None
        self._draft_probs: dict[str, list[torch.Tensor]] = {}

    @_static_shape_args
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _draft_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        assert self.draft_model is not None
        return self.draft_model.compute_logits(hidden)

    def _draft_propose_token(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.speculative_config.draft_sample_method == "greedy":
            return super()._draft_propose_token(hidden)
        logits = self._draft_logits(hidden)
        chunk, self._sampling_chunk = self._sampling_chunk, None
        if chunk is None:
            return logits.argmax(dim=-1).to(torch.int32)
        # DP padding chunks carry no captured target hidden states.
        real_chunk = chunk.hidden_states is not None
        if real_chunk:
            temperatures, top_k, top_p = self.runner._build_padded_sampling_params(
                chunk.start_index, chunk.start_index + chunk.num_reqs, logits
            )
        else:
            temperatures = torch.ones((logits.shape[0], 1), device=logits.device)
            top_k = torch.zeros_like(temperatures, dtype=torch.int32)
            top_p = torch.ones_like(temperatures)
        noise = torch.rand_like(
            logits, dtype=torch.float32, generator=self.runner._get_sampling_generator()
        )
        tokens, probs = self._sample_probabilistic(
            logits,
            temperatures,
            top_k,
            top_p,
            noise,
            self.runner._sampler_applies_topk_topp(),
        )
        if self._draft_replicated:
            tokens = self.runner._sync_one_replicated_draft(tokens)
            probs = self.runner._sync_one_replicated_draft(probs)
        if real_chunk:
            req_ids = self.runner.input_batch.req_ids[
                chunk.start_index : chunk.start_index + chunk.num_reqs
            ]
            for row, req_id in enumerate(req_ids):
                self._draft_probs[req_id].append(probs[row])
        return tokens

    @staticmethod
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _sample_probabilistic(
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        noise: torch.Tensor,
        apply_filter: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        greedy = temperatures <= _SAMPLING_EPS
        scaled = logits.float() / temperatures.float().clamp_min(_SAMPLING_EPS)
        if apply_filter:
            scaled = apply_top_k_top_p(scaled, top_k, top_p)
        uniform = noise.clamp(
            torch.finfo(noise.dtype).tiny, 1 - torch.finfo(noise.dtype).eps
        )
        scores = torch.where(greedy, logits, scaled - torch.log(-torch.log(uniform)))
        tokens = scores.argmax(dim=-1)
        probs = torch.where(
            greedy, F.one_hot(tokens, logits.shape[-1]).float(), scaled.softmax(dim=-1)
        )
        return tokens.to(torch.int32), probs

    def get_draft_probs(
        self,
        req_ids: list[str],
        segment_ids: torch.Tensor,
        group_indices: torch.Tensor,
        draft_lengths: np.ndarray,
    ) -> torch.Tensor | None:
        if self.speculative_config.draft_sample_method == "greedy":
            return None
        k = self.speculative_config.num_speculative_tokens
        vocab = self.speculative_config.draft_model_config.get_vocab_size()
        empty = torch.zeros((k, vocab), device=self.runner.device)
        rows = []
        for row, req_id in enumerate(req_ids):
            values = self._draft_probs.get(req_id)
            if values is None:
                assert draft_lengths[row] == 0, (
                    f"Missing draft probabilities for {req_id}"
                )
            # Prefill and sentinel rows do not enter acceptance.
            rows.append(empty if values is None else torch.stack(values, dim=0))
        rows.append(empty)
        return torch.stack(rows, dim=0)[
            segment_ids.long(), group_indices.long().clamp_max(k - 1)
        ]

    def _dp_lockstep_sharded(self) -> bool:
        # Replicated probabilistic drafts pin both x and q across the target
        # TP group. Idle DP ranks must also emit these collective programs.
        return super()._dp_lockstep_sharded() or (
            self.speculative_config.draft_sample_method == "probabilistic"
            and self.vllm_config.parallel_config.tensor_parallel_size > 1
            and self.runner._dp_lockstep_enabled()
        )

    def _dp_dummy_draft_step(self, dummy: _DpDummyDraft, step: int) -> torch.Tensor:
        if dummy.positions.ndim == 2:
            dummy = dataclasses.replace(
                dummy,
                positions=dummy.positions[0],
                loop_positions=dummy.loop_positions[0],
            )
        if step == 0:
            super()._forward_draft(
                dummy.chunk,
                dummy.input_ids,
                dummy.positions,
                dummy.target_hidden_states,
                0,
                0,
                None,
                dummy.bucket,
            )
            dummy = dataclasses.replace(
                dummy,
                chunk=dataclasses.replace(
                    dummy.chunk,
                    num_reqs=min(dummy.chunk.num_reqs, dummy.loop_bucket),
                ),
                input_ids=dummy.loop_input_ids,
                positions=dummy.loop_positions,
                target_hidden_states=dummy.loop_hidden,
                bucket=dummy.loop_bucket,
            )
        return super()._dp_dummy_draft_step(dummy, step)

    def _prepare_draft_inputs(
        self,
        chunk: DraftChunkInputs,
        sampled_token_ids: list[list[int]] | None,
        discard_sampled_tokens_req_indices: list[int],
        num_rejected_tokens_np: np.ndarray | None,
        scheduler_output: SchedulerOutput,
        next_tokens_device: torch.Tensor | None = None,
        device_seed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray | torch.Tensor]:
        if chunk.position_ids.ndim == 2:
            # An independent text draft uses absolute positions even when the
            # target uses multimodal RoPE axes.
            qsl = chunk.query_start_loc_np[: chunk.num_reqs + 1]
            counts = np.diff(qsl)
            owners = np.repeat(np.arange(chunk.num_reqs), counts)
            owners = np.pad(owners, (0, chunk.input_ids.shape[0] - len(owners)))
            owners_dev = torch.from_numpy(owners).to(self.runner.device)
            starts = torch.from_numpy(qsl[:-1][owners]).to(self.runner.device)
            lengths = torch.from_numpy(counts[owners]).to(self.runner.device)
            offsets = torch.arange(chunk.input_ids.shape[0], device=self.runner.device)
            chunk = dataclasses.replace(
                chunk,
                position_ids=(
                    chunk.attn_ctx.seq_lens[owners_dev] - lengths + offsets - starts
                ),
            )
        seeded_ids, positions, indices, rejected = super()._prepare_draft_inputs(
            chunk,
            sampled_token_ids,
            discard_sampled_tokens_req_indices,
            num_rejected_tokens_np,
            scheduler_output,
            next_tokens_device,
            device_seed,
        )
        assert chunk.hidden_states is not None
        if self.speculative_config.draft_sample_method == "probabilistic":
            self._draft_probs = {
                rid: values
                for rid, values in self._draft_probs.items()
                if rid in self.runner.requests
            }
            for req_id in self.runner.input_batch.req_ids[
                chunk.start_index : chunk.start_index + chunk.num_reqs
            ]:
                self._draft_probs[req_id] = []
        # Replay the unshifted target window into independent KV. Rejected
        # suffix entries remain unreachable through the seed's shortened seq_lens.
        super()._forward_draft(
            chunk,
            chunk.input_ids,
            chunk.position_ids,
            chunk.hidden_states,
            step_idx=0,
            seq_lens_delta=0,
            num_rejected_np=None,
            num_tokens_padded=chunk.input_ids.shape[0],
        )
        if isinstance(rejected, torch.Tensor):
            kernel_num_reqs = (
                self.runner.num_reqs_max_model_len
                if chunk.attn_ctx.use_max_model_len
                else self.runner.num_reqs_most_model_len
            )
            assert kernel_num_reqs is not None
            rejected = fit_rows(rejected, kernel_num_reqs)
        return (
            seeded_ids.index_select(0, indices),
            positions.index_select(0, indices) + 1,
            torch.arange(indices.shape[0], device=indices.device, dtype=indices.dtype),
            rejected,
        )

    def _forward_draft(
        self,
        chunk: DraftChunkInputs,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        step_idx: int,
        seq_lens_delta: int,
        num_rejected_np: np.ndarray | None,
        num_tokens_padded: int | None = None,
        num_rejected_dev: torch.Tensor | None = None,
        loop_query_start_loc: torch.Tensor | None = None,
        loop_request_distribution: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.speculative_config.draft_sample_method == "probabilistic":
            self._sampling_chunk = chunk
        return super()._forward_draft(
            chunk,
            input_ids,
            positions,
            target_hidden_states,
            step_idx + 1,
            seq_lens_delta + 1,
            num_rejected_np,
            num_tokens_padded,
            num_rejected_dev,
            loop_query_start_loc,
            loop_request_distribution,
        )
