# Kunlun Omni-Specific Code Blocks Reference

This document catalogs omni-specific logic in the Kunlun worker/model runner stack so it is easier to preserve during upgrades.

> **IMPORTANT**: This document may not be complete or up to date.
>
> - Always inspect current GPU implementations in `vllm_omni/worker/`.
> - Always inspect current Kunlun target files in `vllm_omni/platforms/kunlun/worker/`.
> - The Kunlun files may rely on MRO inheritance instead of local `Omni-new` markers.
> - When discovering new omni-specific blocks during an upgrade, update this document.

## OmniKunlunWorkerBase (`base.py`)

### Class Purpose

`OmniKunlunWorkerBase` inherits from `vllm_xpu.v1.worker.xpu_worker.XPUWorker` while adding omni worker behavior for memory management, profiling, and sleep/wake control.

```python
from vllm_xpu.v1.worker.xpu_worker import XPUWorker as KunlunWorker

class OmniKunlunWorkerBase(KunlunWorker):
    ...
```

### `load_model` - Weight Memory Pool and Synchronization

```python
def load_model(self, *args, **kwargs):
    with self._maybe_get_memory_pool_context("weights"):
        res = super().load_model(*args, **kwargs)
        current_omni_platform.synchronize()
        gc.collect()
        return res
```

Preserve this behavior so sleep mode can offload weights through the allocator pool and so memory state is stable after loading.

### `__init__` - Omni Profiler Setup

```python
profiler_config = self.vllm_config.profiler_config
if profiler_config and profiler_config.profiler == "torch":
    from vllm_omni.profiler import create_omni_profiler

    stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
    worker_name = f"stage{stage_id}_rank{self.rank}"
    self.profiler = create_omni_profiler(
        profiler_config=profiler_config,
        worker_name=worker_name,
        local_rank=self.local_rank,
    )
```

This is omni-specific because profiles must be stage/rank aware for multi-stage models.

### `profile` - Trace Filename Control

```python
from vllm_omni.profiler import OmniTorchProfilerWrapper

if isinstance(self.profiler, OmniTorchProfilerWrapper):
    stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
    filename = profile_prefix or f"stage{stage_id}_rank{self.rank}_{int(time.time())}"
    self.profiler.set_trace_filename(filename)
```

Preserve this so profiling outputs are not overwritten across stages/ranks.

### `determine_available_memory` - Process-Scoped Profiling

```python
with memory_profiling(
    self.init_snapshot,
    weights_memory=int(self.model_runner.model_memory_usage),
) as profile_result:
    self.model_runner.profile_run()

profiled_usage = (
    int(self.model_runner.model_memory_usage)
    + profile_result.torch_peak_increase
    + profile_result.non_torch_increase
)
self.available_kv_cache_memory_bytes = max(0, self.requested_memory - profiled_usage)
```

Preserve this fallback behavior for concurrent initialization and memory accounting.

### `_maybe_get_memory_pool_context` - Sleep Mode Pool

```python
is_sleep_enabled = v1_config_enabled or getattr(self.cache_config, "enable_sleep_mode", False)
if is_sleep_enabled:
    current_omni_platform.synchronize()
    gc.collect()
    from vllm.device_allocator.cumem import CuMemAllocator

    allocator = CuMemAllocator.get_instance()
    return allocator.use_memory_pool(tag=tag)

return nullcontext()
```

This currently uses `CuMemAllocator` because Kunlun XPU follows CUDA-compatible allocator paths in local `vllm_xpu`. If upstream adds a Kunlun-native allocator, migrate this block carefully.

### `sleep` / `wake_up` - Runtime Memory Control

```python
mem_before = current_omni_platform.get_current_memory_usage(self.device)
allocator.sleep(offload_tags=offload_tags)
current_omni_platform.empty_cache()
current_omni_platform.synchronize()
mem_after = current_omni_platform.get_current_memory_usage(self.device)
```

```python
allocator.wake_up(tags)
current_omni_platform.synchronize()
```

Preserve `current_omni_platform` usage rather than replacing with raw CUDA calls.

### `handle_sleep_task` - OmniACK Emission

```python
ack = OmniACK(
    task_id=task.task_id,
    status="SUCCESS",
    stage_id=current_stage_id,
    rank=self.rank,
    freed_bytes=total_freed,
    metadata={
        "source": "omni_platform_audit",
        "total_freed_gib": f"{total_freed / 1024**3:.2f}",
        "rank_residual_gib": f"{residual_gib:.2f}",
    },
)
if hasattr(self, "result_mq") and self.result_mq:
    self.result_mq.put(ack)
```

Preserve this because external orchestration expects structured ACKs for sleep/wake tasks.

---

## OmniKunlunModelRunner (`kunlun_model_runner.py`)

### Class MRO - Kunlun Backend Plus Omni GPU Behavior

```python
class OmniKunlunModelRunner(XPUModelRunner, OmniGPUModelRunner):
    """Omni model runner backed by the Kunlun XPU runner."""
```

This MRO is the main omni-specific design in Kunlun. `XPUModelRunner` keeps Kunlun backend behavior; `OmniGPUModelRunner` supplies omni loading, forward kwargs, multimodal output, and talker behavior.

### `load_model` - Explicit XPU Path

```python
def load_model(self, *args, **kwargs) -> None:
    XPUModelRunner.load_model(self, *args, **kwargs)
```

This intentionally invokes the XPU load path while allowing MRO continuation into omni GPU logic where upstream `super()` calls require it.

### `_dummy_run` - Multimodal Encoder-Only Fast Path

```python
mm_config = self.vllm_config.model_config.multimodal_config
if mm_config and mm_config.mm_encoder_only:
    return torch.tensor([]), torch.tensor([])
```

Preserve this for encoder-only multimodal configurations.

### `_dummy_run` - Attention Metadata Extensions

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

This is inherited from omni GPU behavior and is required for omni-specific attention metadata extensions.

### `_dummy_run` - XPU MoE Router Patches

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
    self.maybe_dummy_run_with_lora(...),
):
    ...
```

This is Kunlun-specific rather than omni-specific, but it must be preserved when reinserting omni logic into a refreshed `_dummy_run`.

### `_dummy_run` - Multimodal Dummy Inputs

```python
model_kwargs = self._init_model_kwargs()
if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
    input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)
    model_kwargs = {
        **model_kwargs,
        **self._dummy_mm_kwargs(num_reqs),
    }
```

Preserve this so multimodal models exercise their input paths during profiling/warmup.

### `_dummy_run` - `has_preprocess` Input Path

```python
elif getattr(getattr(self, "model", None), "has_preprocess", False):
    input_ids = self.input_ids.gpu[:num_tokens_padded]
    inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
```

Preserve this for models that preprocess both IDs and embeddings.

### `_dummy_run` - Talker MTP Warmup

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

Preserve this so Qwen3-Omni-style talker MTP buffers and graph/warmup behavior remain valid.

### `_dummy_run` - Extract Multimodal Outputs

```python
if self.use_aux_hidden_state_outputs:
    hidden_states, _ = outputs
else:
    hidden_states = outputs
hidden_states, _ = self.extract_multimodal_outputs(hidden_states)
```

This keeps dummy-run return values compatible with models that return `OmniOutput` or multimodal side outputs.

### `_dummy_run` - Embed Warmup

```python
if not getattr(self, "_embed_warmed_up", False) and self.supports_mm_inputs:
    self._embed_warmed_up = True
    dummy_ids = self.input_ids.gpu[:1]
    self.model.embed_input_ids(dummy_ids)
    torch.cuda.synchronize()
```

Audit `torch.cuda.synchronize()` when upgrading. Keep it if local vllm-xpu still uses CUDA-compatible synchronization for Kunlun.

---

## KunlunARModelRunner (`kunlun_ar_model_runner.py`)

### Class MRO

```python
class KunlunARModelRunner(OmniKunlunModelRunner, GPUARModelRunner):
    """Kunlun AR runner for thinker/talker stages."""
```

This keeps Kunlun load/forward/dummy-run behavior while inheriting omni AR execute/sample behavior.

### Device Properties

```python
def _init_device_properties(self):
    self.num_sms = None
```

Preserve unless vllm-xpu provides a true Kunlun equivalent.

### Device Sync

```python
def _sync_device(self) -> None:
    current_omni_platform.synchronize()
```

Preserve to avoid hard-coding `torch.cuda.synchronize()` in AR runner logic.

### Inherited AR Omni Behavior to Preserve

If an upgrade requires copying methods out of `GPUARModelRunner`, preserve the following behaviors:

- `OmniKVTransferManager` setup and finished-request KV transfer handling.
- `_resolve_global_request_id` for disaggregated inference.
- Custom request-state updates before/after scheduling changes.
- `extract_multimodal_outputs` after forward.
- `compute_logits(..., sampling_metadata=...)` fallback behavior.
- Hidden-state CPU copy for output payloads.
- `_process_additional_information_updates`.
- `OmniModelRunnerOutput` construction and `kv_extracted_req_ids` propagation.

---

## KunlunGenerationModelRunner (`kunlun_generation_model_runner.py`)

### Class MRO

```python
class KunlunGenerationModelRunner(OmniKunlunModelRunner, GPUGenerationModelRunner):
    """Kunlun generation runner for non-autoregressive generation stages."""
```

This keeps Kunlun load/forward/dummy-run behavior while inheriting omni generation execute/sample behavior.

### Device Properties

```python
def _init_device_properties(self):
    self.num_sms = None
```

Preserve unless vllm-xpu provides a true Kunlun equivalent.

### Device Sync

```python
def _sync_device(self) -> None:
    current_omni_platform.synchronize()
```

### Inherited Generation Omni Behavior to Preserve

If an upgrade requires copying methods out of `GPUGenerationModelRunner`, preserve the following behaviors:

- `_update_request_states` async chunk handling.
- `seq_token_counts` injection for code2wav/output slicing.
- `_run_generation_model` forwarding semantics.
- `extract_multimodal_outputs` after generation forward.
- Tensor/list/dict multimodal output processing.
- Request ID copies to avoid async scheduling mutation.
- `OmniModelRunnerOutput` construction.
- `ec_connector_output` propagation when `supports_mm_inputs`.

---

## Kunlun Workers

### `KunlunARWorker`

```python
class KunlunARWorker(OmniWorkerMixin, OmniKunlunWorkerBase):
    """Kunlun AR worker for thinker/talker stages in Omni models."""

    def init_device(self):
        super().init_device()
        self.model_runner = KunlunARModelRunner(self.vllm_config, self.device)
```

Preserve the MRO and initialization order.

### `KunlunGenerationWorker`

```python
class KunlunGenerationWorker(OmniWorkerMixin, OmniKunlunWorkerBase):
    """Kunlun generation worker for code2wav stages in Omni models."""

    def init_device(self):
        super().init_device()
        self.model_runner = KunlunGenerationModelRunner(self.vllm_config, self.device)
```

Preserve the MRO and initialization order.

---

## `__init__.py` Exports

Current exports to preserve:

```python
__all__ = [
    "KunlunARModelRunner",
    "KunlunARWorker",
    "KunlunGenerationModelRunner",
    "KunlunGenerationWorker",
    "OmniKunlunModelRunner",
]
```

Update this list if new Kunlun worker or runner classes are added.
