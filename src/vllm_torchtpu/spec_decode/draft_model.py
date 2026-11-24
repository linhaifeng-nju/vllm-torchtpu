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

from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer
from vllm_torchtpu.spec_decode.utils import DraftChunkInputs, fit_rows

if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class DraftModelProposer(Eagle3Proposer):
    """A standalone causal model with its own prompt and decode KV cache."""

    def __init__(self, runner: "TPUModelRunner", vllm_config: VllmConfig) -> None:
        super().__init__(runner, vllm_config)
        self.speculative_config.verify_equal_vocab_size_if_draft_model()
        if not self._draft_tp_matches_target:
            raise NotImplementedError(
                "Standalone TPU drafts require draft_tensor_parallel_size to "
                "equal the target tensor_parallel_size."
            )
        if (
            runner._pcp_enabled
            or vllm_config.parallel_config.pipeline_parallel_size != 1
            or vllm_config.parallel_config.data_parallel_size != 1
            or vllm_config.parallel_config.decode_context_parallel_size != 1
            or vllm_config.cache_config.enable_prefix_caching
            or vllm_config.model_config.is_multimodal_model
            or vllm_config.model_config.is_hybrid
            or self.speculative_config.draft_model_config.is_hybrid
            or self.speculative_config.draft_model_config.is_multimodal_model
            or self.speculative_config.use_heterogeneous_vocab
            or self.speculative_config.draft_sample_method != "greedy"
        ):
            raise NotImplementedError(
                "Standalone TPU drafts currently require dense text models, "
                "DP/PP/DCP=1, no PCP, matching vocabularies, greedy proposals and "
                "--no-enable-prefix-caching."
            )

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
