# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kunlun XPU model runner for vLLM-Omni."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices
import vllm_xpu.envs as xenvs
from vllm_xpu.v1.worker.xpu_runner import XPUModelRunner

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

logger = init_logger(__name__)


class OmniKunlunModelRunner(XPUModelRunner, OmniGPUModelRunner):
    """Omni model runner backed by the Kunlun XPU runner.

    Keep ``XPUModelRunner`` first in the MRO so Kunlun-specific overrides such
    as dummy-run fixes, MoE patches, speculative decoding replacements, and
    tracer wrapping remain active. Its ``super()`` calls then continue into
    ``OmniGPUModelRunner``, preserving Omni-specific model loading, forward
    kwargs injection, multimodal output wrapping, and talker MTP handling.
    """

    def load_model(self, *args, **kwargs) -> None:
        """Load the model through the Kunlun XPU runner path.

        ``XPUModelRunner.load_model`` continues through the MRO into
        ``OmniGPUModelRunner.load_model``, so this keeps both Kunlun-specific
        tracer setup and Omni-specific model initialization.
        """
        XPUModelRunner.load_model(self, *args, **kwargs)

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
        uniform_one_token_decode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a Kunlun XPU dummy forward with Omni warmup behavior.

        This is based on ``XPUModelRunner._dummy_run`` and keeps Kunlun fixes
        for query-start padding, XPU MoE router patches, spec-decode metadata,
        uniform one-token decode, and embed warmup. It also folds in the Omni
        GPU dummy-run additions for attention metadata extensions,
        ``has_preprocess`` inputs, talker MTP warmup, and multimodal output
        extraction.
        """
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            return torch.tensor([]), torch.tensor([])

        assert (
            cudagraph_runtime_mode is None
            or cudagraph_runtime_mode.is_valid_runtime_mode()
        )
        assert not uniform_decode or not uniform_one_token_decode

        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        assert num_tokens <= self.max_num_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        elif uniform_one_token_decode:
            num_reqs = num_tokens
            max_query_len = 1
            num_scheduled_tokens_list = [max_query_len] * num_reqs
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
            self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens,
                max_num_scheduled_tokens=max_query_len,
                use_cascade_attn=False,
                allow_microbatching=allow_microbatching,
                force_eager=is_profile
                or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
                force_uniform_decode=uniform_decode,
                force_has_lora=num_active_loras > 0,
                force_num_active_loras=num_active_loras,
                uniform_one_token_decode=uniform_one_token_decode,
                allow_auto_uniform_one_token_decode=not is_graph_capturing,
            )
        )

        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        elif is_graph_capturing:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            self.vllm_config.parallel_config.num_ubatches,
        )
        logger.debug(
            "ubatch_slices: %s, ubatch_slices_padded: %s",
            ubatch_slices,
            ubatch_slices_padded,
        )

        attn_metadata = None
        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
            num_tokens_padded=num_tokens,
            num_reqs_padded=num_reqs_padded,
            num_tokens_unpadded=num_tokens_unpadded,
            ubatch_slices=ubatch_slices_padded,
        )
        spec_decode_common_attn_metadata = None

        with self.synchronize_input_prep():
            if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
                if profile_seq_lens is not None:
                    seq_lens = profile_seq_lens
                elif create_mixed_batch:
                    seq_lens = torch.tensor(
                        [1] * num_decode_tokens + [num_prefill_tokens + 1],
                        dtype=torch.int,
                    )
                else:
                    seq_lens = max_query_len
                self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
                self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
                self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)

                cum_num_tokens = self._get_cumsum_and_arange(
                    num_scheduled_tokens, self.query_pos.np
                )
                self.query_start_loc.np[0] = 0
                self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
                self.query_start_loc.np[num_reqs + 1 :].fill(cum_num_tokens[-1])
                self.query_start_loc.copy_to_gpu()

                self.input_batch.block_table.commit_block_table(num_reqs_padded)

                pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
                attn_metadata, spec_decode_common_attn_metadata = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs_padded,
                    max_query_len=max_query_len,
                    ubatch_slices=(ubatch_slices_padded if pad_attn else ubatch_slices),
                    for_cudagraph_capture=is_graph_capturing,
                    slot_mappings=slot_mappings_by_group,
                    use_spec_decode=self.speculative_config is not None,
                )
                self._maybe_attach_attention_metadata_extensions(
                    attn_metadata=attn_metadata,
                    num_reqs=num_reqs_padded,
                    num_reqs_padded=num_reqs_padded,
                    max_query_len=max_query_len,
                    pad_attn=True,
                    for_cudagraph_capture=is_graph_capturing,
                )

        from vllm_xpu.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias
        from vllm_xpu.model_executor.layers.fused_moe.router.grouped_topk_router import fused_grouped_topk

        with (
            patch(
                "vllm.model_executor.layers.fused_moe.router.grouped_topk_router.fused_grouped_topk",
                fused_grouped_topk,
            ),
            patch(
                "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router.fused_topk_bias",
                fused_topk_bias,
            ),
            self.maybe_dummy_run_with_lora(
                self.lora_config,
                num_scheduled_tokens,
                num_sampled_tokens,
                remove_lora,
                num_active_loras,
            ),
        ):
            assert num_tokens_padded <= self.max_num_tokens
            model_kwargs = self._init_model_kwargs()
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)
                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
                model_kwargs = self._init_model_kwargs()
            elif getattr(getattr(self, "model", None), "has_preprocess", False):
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions[:num_tokens_padded]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )
                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens_padded, None, False
                )

            if ubatch_slices_padded is not None:
                num_tokens_padded = ubatch_slices_padded[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_padded

            with (
                self.maybe_randomize_inputs(input_ids, inputs_embeds),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    ubatch_slices=ubatch_slices_padded,
                    slot_mapping=slot_mappings,
                ),
            ):
                if getattr(self.model, "talker", None) is not None and self.has_talker_mtp:
                    num_tokens_padded_talker_mtp = num_tokens_padded
                    if num_tokens_padded_talker_mtp == self.max_num_tokens:
                        num_tokens_padded_talker_mtp = self.talker_mtp_input_ids.gpu.shape[0]
                    self.talker_mtp(
                        self.talker_mtp_input_ids.gpu[:num_tokens_padded_talker_mtp],
                        self.talker_mtp_inputs_embeds.gpu[:num_tokens_padded_talker_mtp],
                        self.last_talker_hidden.gpu[:num_tokens_padded_talker_mtp],
                        self.text_step.gpu[:num_tokens_padded_talker_mtp],
                    )
                    self.compilation_config.cache_dir = None
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs
            hidden_states, _ = self.extract_multimodal_outputs(hidden_states)

            if (
                not is_profile
                and not is_graph_capturing
                and xenvs.VOCAB_TENSOR_MODEL_PARALLEL_SIZE > 0
            ):
                if self.use_aux_hidden_state_outputs:
                    self.model.compute_logits(outputs[1])
                else:
                    self.model.compute_logits(hidden_states)

            if self.speculative_config and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
            ):
                assert isinstance(
                    self.drafter,
                    EagleProposer | DraftModelProposer | ExtractHiddenStatesProposer,
                )
                assert self.speculative_config is not None
                use_cudagraphs = (
                    (
                        is_graph_capturing
                        and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                    )
                    or (
                        is_graph_capturing
                        and cudagraph_runtime_mode == CUDAGraphMode.FULL
                    )
                    or (
                        not is_graph_capturing
                        and cudagraph_runtime_mode != CUDAGraphMode.NONE
                    )
                ) and not self.speculative_config.enforce_eager

                if (
                    self.compilation_config.cudagraph_specialize_lora
                    and num_active_loras > 0
                ):
                    use_cudagraphs = False

                need_interleaved_logits = (
                    not is_profile
                    and not is_graph_capturing
                    and xenvs.VOCAB_TENSOR_MODEL_PARALLEL_SIZE > 0
                    and not getattr(self.drafter, "use_local_argmax_reduction", False)
                )
                if need_interleaved_logits:
                    self._drafter_dummy_run_interleaved(
                        num_tokens,
                        use_cudagraphs=use_cudagraphs,
                        is_graph_capturing=is_graph_capturing,
                        slot_mappings=slot_mappings,
                        hidden_states=hidden_states,
                    )
                else:
                    self.drafter.dummy_run(
                        num_tokens,
                        use_cudagraphs=use_cudagraphs,
                        is_graph_capturing=is_graph_capturing,
                        slot_mappings=slot_mappings,
                        common_attn_metadata=spec_decode_common_attn_metadata,
                        num_reqs=num_reqs,
                        uniform_decode=uniform_decode,
                        uniform_one_token_decode=uniform_one_token_decode,
                    )

        self._register_layerwise_nvtx_hooks()

        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        logit_indices_device = torch.from_numpy(logit_indices).to(
            self.device, non_blocking=True
        )

        if not getattr(self, "_embed_warmed_up", False) and self.supports_mm_inputs:
            self._embed_warmed_up = True
            dummy_ids = self.input_ids.gpu[:1]
            self.model.embed_input_ids(dummy_ids)
            torch.cuda.synchronize()

        return hidden_states, hidden_states[logit_indices_device]

    @torch.inference_mode()
    def _drafter_dummy_run_interleaved(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> None:
        """Run drafter warmup with logits between forwards for XPU vocab TP."""
        drafter = self.drafter
        num_spec_tokens = self.speculative_config.num_speculative_tokens
        only_one_forward_pass = is_graph_capturing or drafter.parallel_drafting

        for fwd_idx in range(1 if only_one_forward_pass else num_spec_tokens):
            if fwd_idx <= 1:
                result = drafter._determine_batch_execution_and_padding(
                    num_tokens, use_cudagraphs=use_cudagraphs
                )
                cudagraph_runtime_mode = result[0]
                num_input_tokens = result[1]
                num_tokens_across_dp = result[2]

            if (
                drafter._draft_attn_layer_names
                and slot_mappings is not None
                and next(iter(drafter._draft_attn_layer_names)) in slot_mappings
            ):
                slot_mapping_dict = drafter._get_slot_mapping(num_input_tokens)
            else:
                slot_mapping_dict = slot_mappings or {}

            with set_forward_context(
                None,
                drafter.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping_dict,
            ):
                if drafter.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = drafter.inputs_embeds[:num_input_tokens]
                else:
                    input_ids = drafter.input_ids[:num_input_tokens]
                    inputs_embeds = None

                kwargs = dict(
                    input_ids=input_ids,
                    positions=drafter._get_positions(num_input_tokens),
                    inputs_embeds=inputs_embeds,
                )
                if drafter.pass_hidden_states_to_model:
                    kwargs["hidden_states"] = drafter.hidden_states[:num_input_tokens]
                draft_output = drafter.model(**kwargs)

            if draft_output is not None:
                if isinstance(draft_output, tuple):
                    draft_hidden_for_logits = draft_output[0][:1]
                else:
                    draft_hidden_for_logits = draft_output[:1]
            else:
                assert hidden_states is not None
                draft_hidden_for_logits = hidden_states[:1]
            drafter.model.compute_logits(draft_hidden_for_logits)

    def _model_forward(self, *args, **kwargs):
        from vllm_xpu.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias
        from vllm_xpu.model_executor.layers.fused_moe.router.grouped_topk_router import fused_grouped_topk

        with (
            patch(
                "vllm.model_executor.layers.fused_moe.router.grouped_topk_router.fused_grouped_topk",
                fused_grouped_topk,
            ),
            patch(
                "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router.fused_topk_bias",
                fused_topk_bias,
            ),
        ):
            return super()._model_forward(*args, **kwargs)
