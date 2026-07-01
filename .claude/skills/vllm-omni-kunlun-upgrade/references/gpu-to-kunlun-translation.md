# GPU to Kunlun Translation Patterns

This document provides a quick reference for translating GPU model runner patterns to Kunlun XPU equivalents when preserving omni-specific logic.

Kunlun in this repo is backed by `vllm_xpu`. Some code paths are CUDA-compatible, but do not assume every GPU API is semantically correct for Kunlun. Prefer the local `vllm_xpu` implementation and `current_omni_platform` abstractions.

## Import Translations

### Base Model Runner

```python
# GPU / Omni reference
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

# Kunlun target
from vllm_xpu.v1.worker.xpu_runner import XPUModelRunner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
```

Kunlun target class:

```python
class OmniKunlunModelRunner(XPUModelRunner, OmniGPUModelRunner):
    ...
```

Keep `XPUModelRunner` first in the MRO.

### Base Worker

```python
# Kunlun backend worker
from vllm_xpu.v1.worker.xpu_worker import XPUWorker as KunlunWorker

# Omni platform helpers
from vllm_omni.platforms import current_omni_platform
```

Kunlun target class:

```python
class OmniKunlunWorkerBase(KunlunWorker):
    ...
```

### AR Runner

```python
# GPU / Omni reference
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

# Kunlun target
from vllm_omni.platforms.kunlun.worker.kunlun_model_runner import OmniKunlunModelRunner
```

Kunlun target class:

```python
class KunlunARModelRunner(OmniKunlunModelRunner, GPUARModelRunner):
    ...
```

### Generation Runner

```python
# GPU / Omni reference
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner

# Kunlun target
from vllm_omni.platforms.kunlun.worker.kunlun_model_runner import OmniKunlunModelRunner
```

Kunlun target class:

```python
class KunlunGenerationModelRunner(OmniKunlunModelRunner, GPUGenerationModelRunner):
    ...
```

### Worker Mixins

```python
from vllm_omni.worker.mixins import OmniWorkerMixin
from vllm_omni.platforms.kunlun.worker.base import OmniKunlunWorkerBase
```

Worker target classes:

```python
class KunlunARWorker(OmniWorkerMixin, OmniKunlunWorkerBase):
    ...

class KunlunGenerationWorker(OmniWorkerMixin, OmniKunlunWorkerBase):
    ...
```

## Context Manager Translations

### Forward Context Setup

```python
# GPU / current Kunlun path
from vllm.forward_context import set_forward_context

with set_forward_context(
    attn_metadata,
    self.vllm_config,
    num_tokens=num_tokens_padded,
    num_tokens_across_dp=num_tokens_across_dp,
    cudagraph_runtime_mode=cudagraph_runtime_mode,
    batch_descriptor=batch_desc,
    ubatch_slices=ubatch_slices_padded,
    slot_mapping=slot_mappings,
):
    outputs = self.model(...)
```

Do not translate this to `set_ascend_forward_context`. If local `vllm_xpu` introduces a Kunlun-specific forward context, copy the latest `XPUModelRunner` structure and reinsert omni logic.

### Memory Pool Context

```python
# Kunlun worker base
with self._maybe_get_memory_pool_context("weights"):
    res = super().load_model(*args, **kwargs)
    current_omni_platform.synchronize()
    gc.collect()
    return res
```

Keep this pattern to support sleep-mode weight memory management.

## Device Operations

### Synchronization

```python
# GPU
import torch
torch.cuda.synchronize()

# Kunlun preferred path in vllm-omni
from vllm_omni.platforms import current_omni_platform
current_omni_platform.synchronize()
```

Some `torch.cuda` calls may remain because `vllm_xpu` exposes CUDA-compatible APIs. Audit each one before changing it.

### Memory Queries

```python
# Prefer omni platform abstraction
mem = current_omni_platform.get_current_memory_usage(self.device)
free = current_omni_platform.get_free_memory(self.device)
total = current_omni_platform.get_device_total_memory()
```

Fallbacks to `torch.cuda.get_device_properties` should be explicit and guarded.

### Empty Cache

```python
# Kunlun preferred path
current_omni_platform.empty_cache()
current_omni_platform.synchronize()
```

## Graph and Capture Patterns

### Graph Wrapper

Kunlun currently inherits vLLM/XPU behavior and may expose CUDA-compatible graph concepts through `CUDAGraphMode`.

```python
# Allowed only if local vllm-xpu uses this compatibility path
from vllm.config import CUDAGraphMode
```

### Runtime Mode Naming

Current Kunlun code still uses `cudagraph_runtime_mode` because it follows vLLM/XPU APIs.

```python
cudagraph_runtime_mode: CUDAGraphMode | None = None
```

Do not rename this to `aclgraph_runtime_mode` in Kunlun code.

## Attention Metadata

### Building Attention Metadata

Use latest `XPUModelRunner` as the source of truth.

```python
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
```

Preserve omni metadata extension calls if present:

```python
self._maybe_attach_attention_metadata_extensions(
    attn_metadata=attn_metadata,
    num_reqs=num_reqs_padded,
    num_reqs_padded=num_reqs_padded,
    max_query_len=max_query_len,
    pad_attn=True,
    for_cudagraph_capture=is_graph_capturing,
)
```

Do not introduce `AscendAttentionState` into Kunlun code.

## XPU MoE Router Patches

Kunlun `_dummy_run` currently patches vLLM MoE router functions with vllm-xpu implementations:

```python
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
    ...
```

Preserve this when syncing `_dummy_run` unless latest `vllm_xpu` removes or replaces it.

## Sampling

Kunlun AR/generation currently inherits sampling behavior through omni GPU runner classes and vLLM/XPU base classes.

```python
# Keep inherited sampler path unless local vllm-xpu adds a custom XPU sampler.
```

If local vllm-xpu adds a Kunlun-specific sampler, update the base runner from `XPUModelRunner` first and then reinsert omni output handling.

## Input Batch

Use whatever input batch class and buffers latest `XPUModelRunner` uses.

Relevant current buffer names in Kunlun `_dummy_run` include:

```python
self.input_batch.block_table
self.query_pos
self.query_start_loc
self.seq_lens
self.input_ids
self.inputs_embeds
self.positions
self.mrope_positions
self.xdrope_positions
```

## Omni Behavior Translations

### Talker MTP Dummy Forward

```python
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
```

### Multimodal Output Extraction

```python
if self.use_aux_hidden_state_outputs:
    hidden_states, _ = outputs
else:
    hidden_states = outputs
hidden_states, _ = self.extract_multimodal_outputs(hidden_states)
```

### Multimodal Dummy Inputs

```python
if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
    input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)
    model_kwargs = {
        **model_kwargs,
        **self._dummy_mm_kwargs(num_reqs),
    }
```

### Preprocess Models

```python
elif getattr(getattr(self, "model", None), "has_preprocess", False):
    input_ids = self.input_ids.gpu[:num_tokens_padded]
    inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
```

## Quick Reference Table

| Feature | GPU | Kunlun |
|---------|-----|--------|
| Base runner | `GPUModelRunner` / `OmniGPUModelRunner` | `XPUModelRunner` + `OmniGPUModelRunner` |
| Base worker | GPU worker base | `XPUWorker as KunlunWorker` + `OmniKunlunWorkerBase` |
| Forward context | `set_forward_context` | `set_forward_context` unless vllm-xpu changes |
| Runtime mode param | `cudagraph_runtime_mode` | `cudagraph_runtime_mode` |
| Device sync | `torch.cuda.synchronize()` | Prefer `current_omni_platform.synchronize()` |
| Memory allocator | CUDA allocator | Current code uses `CuMemAllocator`; migrate if vllm-xpu adds native allocator |
| Graph wrapper | `CUDAGraphWrapper` | Use only local vllm-xpu-supported graph path; never `ACLGraphWrapper` |
| Attention state | vLLM attention metadata | Latest `XPUModelRunner` attention metadata |
| Sampler | vLLM sampler | Inherited vLLM/XPU sampler path; never `AscendSampler` |
| MoE router | vLLM router | Patch to `vllm_xpu` fused router functions if still required |
| Omni output | `OmniModelRunnerOutput` | Preserve through GPU AR/generation inheritance or explicit override |
