# Kunlun Model Runner Upgrade Workflow Checklist

> **Note**: Reference documents may not be complete. Always inspect the current `vllm_xpu` backend and grep the GPU implementations for omni-specific behavior before editing Kunlun files.

## Pre-Upgrade Preparation

### 1. Version Information

- [ ] Identify current vllm-omni branch/version: `_________`
- [ ] Identify local vllm-xpu Kunlun backend version/source: `_________`
- [ ] Identify target vllm version: `_________`
- [ ] Identify last release/date used for GPU worker comparison: `_________`

### 2. Locate Backend Sources

```bash
# In vllm-omni, confirm Kunlun integration points
grep -R "vllm_xpu\|XPUModelRunner\|XPUWorker" vllm_omni/platforms/kunlun setup.py vllm_omni/platforms/__init__.py

# In local vllm-xpu source, locate the latest worker/runner definitions
grep -R "class XPUModelRunner\|class XPUWorker" <vllm-xpu-source>
```

### 3. Gather Git History

```bash
# GPU-side omni changes
cd /home/work/cosmos3_adapter/vllm-omni
git log --oneline -- vllm_omni/worker/

# Kunlun target changes
git log --oneline -- vllm_omni/platforms/kunlun/worker/
```

### 4. Backup Current Files

- [ ] Create a backup or rely on git before editing:

```bash
cp -r vllm_omni/platforms/kunlun/worker vllm_omni/platforms/kunlun/worker.backup
```

---

## OmniKunlunWorkerBase (`base.py`)

### Read and Understand

- [ ] Read current `vllm_omni/platforms/kunlun/worker/base.py`
- [ ] Read latest `vllm_xpu.v1.worker.xpu_worker.XPUWorker`
- [ ] Read `vllm_omni/worker/base.py` and `vllm_omni/worker/mixins.py` if worker behavior changed

### Method: `load_model`

- [ ] Keep `_maybe_get_memory_pool_context("weights")`
- [ ] Keep `super().load_model(*args, **kwargs)` inside the memory pool context
- [ ] Keep `current_omni_platform.synchronize()` after loading
- [ ] Keep `gc.collect()` after synchronization
- [ ] Verify upstream `XPUWorker.load_model` did not add required arguments or return behavior

### Method: `__init__`

- [ ] Keep `super().__init__(*args, **kwargs)` first
- [ ] Keep omni profiler creation when `profiler_config.profiler == "torch"`
- [ ] Preserve stage/rank naming: `stage{stage_id}_rank{self.rank}`
- [ ] Verify profiler imports still match `vllm_omni.profiler`

### Method: `profile`

- [ ] Keep explicit error when profiler is disabled
- [ ] Keep `OmniTorchProfilerWrapper.set_trace_filename(...)`
- [ ] Preserve `profile_prefix` override
- [ ] Verify start/stop semantics match upstream worker expectations

### Method: `determine_available_memory`

- [ ] Preserve `kv_cache_memory_bytes` fast path
- [ ] Preserve `memory_profiling(...)` around `self.model_runner.profile_run()`
- [ ] Use `current_omni_platform` for memory where possible
- [ ] Verify `model_memory_usage`, `requested_memory`, and `available_kv_cache_memory_bytes` still exist upstream

### Method: `_maybe_get_memory_pool_context`

- [ ] Preserve `enable_sleep_mode` checks from both v1 config and cache config
- [ ] Keep synchronization and GC before allocator use
- [ ] Keep `CuMemAllocator` only if local vllm-xpu still uses CUDA-compatible allocator paths
- [ ] Switch to a Kunlun/XPU-native allocator if upstream vllm-xpu introduces one

### Methods: `sleep` / `wake_up`

- [ ] Keep `current_omni_platform.get_current_memory_usage`
- [ ] Keep `current_omni_platform.empty_cache()` and `synchronize()`
- [ ] Verify allocator tags match current sleep-mode conventions
- [ ] Avoid raw `torch.cuda` unless the local vllm-xpu path requires it

### Methods: `handle_sleep_task` / `handle_wake_task`

- [ ] Preserve `OmniSleepTask`, `OmniWakeTask`, and `OmniACK` handling
- [ ] Preserve dict-to-task conversion
- [ ] Preserve distributed all-reduce and barrier handling
- [ ] Preserve rank-0-only ACK emission behavior
- [ ] Preserve `result_mq.put(ack)` when available
- [ ] Verify error path returns an `OmniACK` with `status="ERROR"`

---

## OmniKunlunModelRunner (`kunlun_model_runner.py`)

### Read and Understand

- [ ] Read current `kunlun_model_runner.py`
- [ ] Read latest local `vllm_xpu.v1.worker.xpu_runner.XPUModelRunner`
- [ ] Read latest `vllm_omni/worker/gpu_model_runner.py`

### Class MRO

- [ ] Keep `class OmniKunlunModelRunner(XPUModelRunner, OmniGPUModelRunner)` unless there is a strong upstream reason to change it
- [ ] Confirm `XPUModelRunner` remains first so Kunlun backend fixes take precedence
- [ ] Confirm `OmniGPUModelRunner` remains in the MRO so omni logic is inherited

### Method: `load_model`

- [ ] Keep direct `XPUModelRunner.load_model(self, *args, **kwargs)` if upstream MRO behavior still composes correctly
- [ ] Verify Kunlun tracer setup, dummy-run fixes, and model initialization are preserved
- [ ] Verify omni model loading, talker setup, and multimodal behavior still execute through the MRO

### Method: `_dummy_run`

- [ ] Copy structural changes from latest `XPUModelRunner._dummy_run`
- [ ] Preserve max token/request calculation and padding behavior
- [ ] Preserve `_determine_batch_execution_and_padding` call
- [ ] Preserve `maybe_create_ubatch_slices`
- [ ] Preserve `_get_slot_mappings` and `_build_attention_metadata`
- [ ] Preserve `_maybe_attach_attention_metadata_extensions`
- [ ] Preserve XPU MoE router patches from `vllm_xpu.model_executor.layers.fused_moe`
- [ ] Preserve `maybe_dummy_run_with_lora`
- [ ] Preserve multimodal dummy input preparation
- [ ] Preserve `has_preprocess` input path
- [ ] Preserve talker MTP dummy forward block
- [ ] Preserve `self.compilation_config.cache_dir = None` after talker MTP warmup
- [ ] Preserve `hidden_states, _ = self.extract_multimodal_outputs(hidden_states)`
- [ ] Preserve speculative drafter dummy run handling
- [ ] Preserve embed warmup if `supports_mm_inputs`
- [ ] Audit any `torch.cuda.synchronize()` call and keep only if vllm-xpu expects CUDA-compatible API

### Forward Context

- [ ] Keep `set_forward_context` unless local vllm-xpu changes to a Kunlun-specific context API
- [ ] Preserve arguments required by latest vllm core/vllm-xpu: `num_tokens`, `num_tokens_across_dp`, `cudagraph_runtime_mode`, `batch_descriptor`, `ubatch_slices`, `slot_mapping`
- [ ] Do not replace with `set_ascend_forward_context`

### Future Method: `_model_forward`

If upstream vllm-xpu adds or changes `_model_forward`:

- [ ] Use `XPUModelRunner` as the structure baseline
- [ ] Reinsert `_build_model_kwargs_extra()` if GPU omni logic requires it
- [ ] Reinsert `OmniOutput` wrapping if needed
- [ ] Preserve `_omni_last_model_output` caching if sample paths depend on it
- [ ] Preserve `extract_multimodal_outputs`
- [ ] Do not introduce Ascend graph/context update logic

---

## KunlunARModelRunner (`kunlun_ar_model_runner.py`)

### Read and Understand

- [ ] Read current `kunlun_ar_model_runner.py`
- [ ] Read latest `vllm_omni/worker/gpu_ar_model_runner.py`
- [ ] Check whether inherited `GPUARModelRunner` behavior still composes with `OmniKunlunModelRunner`

### Class MRO

- [ ] Keep `class KunlunARModelRunner(OmniKunlunModelRunner, GPUARModelRunner)` if possible
- [ ] Prefer MRO composition over copying large GPU AR methods
- [ ] Add explicit overrides only for Kunlun-specific device/backend behavior

### Method: `_init_device_properties`

- [ ] Keep `self.num_sms = None` unless local vllm-xpu introduces a real Kunlun equivalent

### Method: `_sync_device`

- [ ] Keep `current_omni_platform.synchronize()`
- [ ] Do not replace with raw `torch.cuda.synchronize()` unless required by local vllm-xpu

### If Overriding `execute_model`

- [ ] Preserve `OmniKVTransferManager` behavior if present in GPU AR logic
- [ ] Preserve request-state updates and scheduling semantics
- [ ] Preserve hidden-state and multimodal output extraction
- [ ] Preserve compute-logits fallback for `sampling_metadata`
- [ ] Preserve `ExecuteModelState` fields if the method returns async state

### If Overriding `sample_tokens`

- [ ] Preserve `kv_extracted_req_ids` propagation
- [ ] Preserve hidden-state CPU copy when required by omni output handling
- [ ] Preserve `_process_additional_information_updates`
- [ ] Preserve `OmniModelRunnerOutput` construction
- [ ] Preserve `pooler_output` behavior for non-text engine output types

---

## KunlunGenerationModelRunner (`kunlun_generation_model_runner.py`)

### Read and Understand

- [ ] Read current `kunlun_generation_model_runner.py`
- [ ] Read latest `vllm_omni/worker/gpu_generation_model_runner.py`
- [ ] Check whether inherited `GPUGenerationModelRunner` behavior still composes with `OmniKunlunModelRunner`

### Class MRO

- [ ] Keep `class KunlunGenerationModelRunner(OmniKunlunModelRunner, GPUGenerationModelRunner)` if possible
- [ ] Prefer MRO composition over copying large GPU generation methods
- [ ] Add explicit overrides only for Kunlun-specific device/backend behavior

### Method: `_init_device_properties`

- [ ] Keep `self.num_sms = None` unless local vllm-xpu introduces a real Kunlun equivalent

### Method: `_sync_device`

- [ ] Keep `current_omni_platform.synchronize()`

### If Overriding `_update_request_states`

- [ ] Preserve async chunk behavior
- [ ] Preserve request IDs and scheduling state expected by generation output slicing

### If Overriding `execute_model`

- [ ] Preserve async chunk update logic
- [ ] Preserve `seq_token_counts` injection for code2wav/output slicing
- [ ] Preserve `_run_generation_model` call semantics
- [ ] Preserve multimodal output extraction
- [ ] Preserve `ExecuteModelState` compatibility with AR runner if shared

### If Overriding `sample_tokens`

- [ ] Preserve tensor/list/dict multimodal output handling
- [ ] Preserve request ID copies to avoid async scheduling mutation
- [ ] Preserve `OmniModelRunnerOutput` construction
- [ ] Preserve `ec_connector_output` when `supports_mm_inputs`

---

## Kunlun Workers

### `kunlun_ar_worker.py`

- [ ] Keep `class KunlunARWorker(OmniWorkerMixin, OmniKunlunWorkerBase)`
- [ ] Keep `super().init_device()` first
- [ ] Keep `self.model_runner = KunlunARModelRunner(self.vllm_config, self.device)`

### `kunlun_generation_worker.py`

- [ ] Keep `class KunlunGenerationWorker(OmniWorkerMixin, OmniKunlunWorkerBase)`
- [ ] Keep `super().init_device()` first
- [ ] Keep `self.model_runner = KunlunGenerationModelRunner(self.vllm_config, self.device)`

### `__init__.py`

- [ ] Export `KunlunARModelRunner`
- [ ] Export `KunlunARWorker`
- [ ] Export `KunlunGenerationModelRunner`
- [ ] Export `KunlunGenerationWorker`
- [ ] Export `OmniKunlunModelRunner`

---

## Post-Upgrade Validation

### Syntax Validation

- [ ] `python -m py_compile vllm_omni/platforms/kunlun/worker/base.py`
- [ ] `python -m py_compile vllm_omni/platforms/kunlun/worker/kunlun_model_runner.py`
- [ ] `python -m py_compile vllm_omni/platforms/kunlun/worker/kunlun_ar_model_runner.py`
- [ ] `python -m py_compile vllm_omni/platforms/kunlun/worker/kunlun_generation_model_runner.py`
- [ ] `python -m py_compile vllm_omni/platforms/kunlun/worker/kunlun_ar_worker.py`
- [ ] `python -m py_compile vllm_omni/platforms/kunlun/worker/kunlun_generation_worker.py`

### Import Validation

- [ ] `python -c "from vllm_omni.platforms.kunlun.worker import *"`
- [ ] `python -c "from vllm_omni.platforms.kunlun.worker.kunlun_model_runner import OmniKunlunModelRunner"`
- [ ] `python -c "from vllm_omni.platforms.kunlun.worker.kunlun_ar_model_runner import KunlunARModelRunner"`
- [ ] `python -c "from vllm_omni.platforms.kunlun.worker.kunlun_generation_model_runner import KunlunGenerationModelRunner"`

### Code Review

- [ ] `XPUModelRunner` remains first in base runner MRO
- [ ] `OmniWorkerMixin` remains in worker MRO
- [ ] Raw `torch.cuda` calls were audited
- [ ] Sleep/wake behavior still emits ACKs correctly
- [ ] No large GPU method was copied when MRO inheritance was sufficient

---

## Git Commit

### Commit Message Template

```text
[Kunlun] Upgrade model runners to align with vllm-xpu

- Update OmniKunlunWorkerBase with latest Kunlun worker behavior
- Update OmniKunlunModelRunner with latest XPUModelRunner behavior
- Preserve GPU-side omni logic through MRO and targeted overrides
- Preserve sleep/wake, profiling, talker, and multimodal output behavior

Changes from vllm-xpu:
- <list key changes>

Changes synced from GPU omni:
- <list key changes>
```

### Files to Stage

- [ ] `vllm_omni/platforms/kunlun/worker/base.py`
- [ ] `vllm_omni/platforms/kunlun/worker/kunlun_model_runner.py`
- [ ] `vllm_omni/platforms/kunlun/worker/kunlun_ar_model_runner.py`
- [ ] `vllm_omni/platforms/kunlun/worker/kunlun_generation_model_runner.py`
- [ ] `vllm_omni/platforms/kunlun/worker/kunlun_ar_worker.py`
- [ ] `vllm_omni/platforms/kunlun/worker/kunlun_generation_worker.py`
- [ ] Any updated skill reference files

---

## Troubleshooting

### Import Errors

- Check whether `vllm_xpu` is installed or available on `PYTHONPATH`
- Check whether `vllm_xpu.v1.worker.xpu_runner` or `xpu_worker` paths changed
- Check whether Kunlun platform probing in `setup.py` or `vllm_omni/platforms/__init__.py` still matches local vllm-xpu

### Type Errors

- Check method signatures against latest `XPUModelRunner`
- Check MRO order with `OmniKunlunModelRunner.__mro__`
- Check whether inherited GPU AR/generation methods expect attributes initialized by `OmniKunlunModelRunner`

### Runtime Errors

- Enable debug logging: `export VLLM_LOGGING_LEVEL=DEBUG`
- Try eager mode if graph capture fails
- Check Kunlun/XPU fused MoE router patches
- Check `current_omni_platform` memory and synchronization methods

### Performance Regression

- Compare with previous Kunlun build on the same model and batch shape
- Check whether XPU graph/cudagraph-compatible paths still trigger
- Check whether MoE router patches are still applied
- Check whether extra overrides bypassed upstream XPU optimized paths
