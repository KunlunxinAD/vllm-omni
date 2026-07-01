---
name: vllm-omni-kunlun-model-runner-upgrade
description: "Upgrade vllm-omni Kunlun XPU model runners and workers (OmniKunlunModelRunner, KunlunARModelRunner, KunlunGenerationModelRunner, OmniKunlunWorkerBase) to align with the latest vllm-xpu Kunlun backend while preserving omni-specific logic. Use this whenever modifying, syncing, reviewing, or upgrading vllm_omni/platforms/kunlun/worker code."
---

# vLLM-Omni Kunlun Model Runner Upgrade Skill

## Overview

This skill guides upgrades of vllm-omni's Kunlun XPU worker and model runner stack. The goal is to align the Kunlun implementation with the latest local `vllm-xpu` Kunlun backend while preserving vllm-omni behavior for multimodal models, thinker/talker stages, generation stages, sleep/wake memory control, profiling, and multimodal output packaging.

Use the local `vllm-xpu` backend as the execution baseline, use vllm-omni GPU workers/runners as the authoritative source for omni-specific behavior, and use the existing Kunlun files as the migration target.

## File Structure

### Kunlun Worker Files

```text
vllm-omni/vllm_omni/platforms/kunlun/worker/
├── __init__.py
├── base.py                           # OmniKunlunWorkerBase
├── kunlun_model_runner.py            # OmniKunlunModelRunner
├── kunlun_ar_model_runner.py         # KunlunARModelRunner
├── kunlun_ar_worker.py               # KunlunARWorker
├── kunlun_generation_model_runner.py # KunlunGenerationModelRunner
└── kunlun_generation_worker.py       # KunlunGenerationWorker
```

### GPU Reference Files for Omni Logic

```text
vllm-omni/vllm_omni/worker/
├── gpu_model_runner.py               # OmniGPUModelRunner
├── gpu_ar_model_runner.py            # GPUARModelRunner
├── gpu_ar_worker.py
├── gpu_generation_model_runner.py    # GPUGenerationModelRunner
├── gpu_generation_worker.py
├── mixins.py                         # OmniWorkerMixin
├── base.py
└── gpu_memory_utils.py
```

### Kunlun Backend Reference Files

The Kunlun implementation inherits from `vllm_xpu`. Locate the exact local files before editing; module paths may change between `vllm-xpu` versions.

Current known imports in this repo:

```python
from vllm_xpu.v1.worker.xpu_runner import XPUModelRunner
from vllm_xpu.v1.worker.xpu_worker import XPUWorker as KunlunWorker
from vllm_xpu.platforms.kunlun import XPU3Platform
```

Search local source for:

```text
XPUModelRunner
XPUWorker
xpu_runner.py
xpu_worker.py
vllm_xpu.platforms.kunlun
```

## Reference Priority

1. Latest local `vllm_xpu` Kunlun/XPU runner and worker implementation.
2. vllm-omni GPU runner and worker files for omni-specific behavior.
3. Existing `vllm_omni/platforms/kunlun/worker` files as migration targets.


## Inheritance Hierarchy

The hierarchy is multiple-inheritance-based. Each Kunlun class composes a backend baseline (`vllm_xpu`) with an omni baseline (`vllm-omni/worker`). The diagrams below show actual parent-class relationships and the order they appear in each class's MRO.

### Upstream Base Classes

```text
vllm core                       vllm_xpu                          vllm-omni
─────────                       ──────────────────                ─────────────────────────────
GPUModelRunner          ──►     XPUModelRunner                    OmniGPUModelRunner
                                (vllm_xpu/v1/                      (vllm_omni/worker/
                                 worker/xpu_runner.py)             gpu_model_runner.py)
                                                                            ▲
                                                                            │
                                                                  GPUARModelRunner
                                                                  GPUGenerationModelRunner
                                                                  (also mix in
                                                                   OmniConnectorModelRunnerMixin)

Worker (vllm.v1.worker.gpu_worker)
        ▲
        │
XPUWorker (vllm_xpu/v1/worker/xpu_worker.py)                      OmniWorkerMixin
                                                                  (vllm_omni/worker/mixins.py)
```

Confirmed from source:

```python
# vllm_xpu/v1/worker/xpu_runner.py
class XPUModelRunner(GPUModelRunner): ...

# vllm_xpu/v1/worker/xpu_worker.py
class XPUWorker(Worker): ...                       # Worker = vllm.v1.worker.gpu_worker.Worker

# vllm_omni/worker/gpu_model_runner.py
class OmniGPUModelRunner(GPUModelRunner): ...

# vllm_omni/worker/gpu_ar_model_runner.py
class GPUARModelRunner(OmniGPUModelRunner, OmniConnectorModelRunnerMixin): ...

# vllm_omni/worker/gpu_generation_model_runner.py
class GPUGenerationModelRunner(OmniGPUModelRunner, OmniConnectorModelRunnerMixin): ...
```

### Kunlun Model Runner Hierarchy

```text
              XPUModelRunner                    OmniGPUModelRunner
              (vllm_xpu)                        (vllm-omni)
                    └────────────┬───────────────────┘
                                 ▼
                       OmniKunlunModelRunner
                       (XPUModelRunner, OmniGPUModelRunner)
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
      KunlunARModelRunner                 KunlunGenerationModelRunner
      (OmniKunlunModelRunner,             (OmniKunlunModelRunner,
       GPUARModelRunner)                   GPUGenerationModelRunner)
```

Confirmed from source:

```python
class OmniKunlunModelRunner(XPUModelRunner, OmniGPUModelRunner): ...
class KunlunARModelRunner(OmniKunlunModelRunner, GPUARModelRunner): ...
class KunlunGenerationModelRunner(OmniKunlunModelRunner, GPUGenerationModelRunner): ...
```

MRO rules to keep:

- `XPUModelRunner` is the first parent of `OmniKunlunModelRunner` so Kunlun backend behavior (dummy-run, MoE router patches, attention metadata, spec-decode, drafter handling) takes precedence over generic GPU paths.
- `OmniGPUModelRunner` is the second parent so omni base behavior (multimodal output extraction, talker MTP, omni model kwargs) is reachable through `super()`.
- `OmniKunlunModelRunner` is the first parent of AR/Generation runners so Kunlun overrides apply before omni AR/Generation execute/sample logic.
- `GPUARModelRunner` and `GPUGenerationModelRunner` are second parents so omni execute/sample paths are inherited rather than copied.
- `GPUARModelRunner` and `GPUGenerationModelRunner` themselves inherit `OmniGPUModelRunner` plus `OmniConnectorModelRunnerMixin`. C3 linearization keeps `OmniGPUModelRunner` reachable from one branch only, which is why `OmniKunlunModelRunner` already inherits it directly and the Kunlun AR/Generation classes do not need to.

### Kunlun Worker Hierarchy

```text
                XPUWorker                          OmniWorkerMixin
                (vllm_xpu, alias KunlunWorker)     (vllm-omni)
                    │                                    │
                    ▼                                    │
          OmniKunlunWorkerBase                           │
          (KunlunWorker)                                 │
                    │                                    │
                    └────────────────┬───────────────────┘
                                     ▼
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
            KunlunARWorker                    KunlunGenerationWorker
            (OmniWorkerMixin,                 (OmniWorkerMixin,
             OmniKunlunWorkerBase)             OmniKunlunWorkerBase)
```

Confirmed from source:

```python
# vllm_omni/platforms/kunlun/worker/base.py
from vllm_xpu.v1.worker.xpu_worker import XPUWorker as KunlunWorker
class OmniKunlunWorkerBase(KunlunWorker): ...

# vllm_omni/platforms/kunlun/worker/kunlun_ar_worker.py
class KunlunARWorker(OmniWorkerMixin, OmniKunlunWorkerBase): ...

# vllm_omni/platforms/kunlun/worker/kunlun_generation_worker.py
class KunlunGenerationWorker(OmniWorkerMixin, OmniKunlunWorkerBase): ...
```

MRO rules to keep:

- `KunlunWorker` is just an alias for `vllm_xpu.v1.worker.xpu_worker.XPUWorker`; do not introduce a separate class.
- `OmniWorkerMixin` is the first parent of each concrete Kunlun worker so its omni overrides win attribute lookup.
- `OmniKunlunWorkerBase` is the second parent so `XPUWorker` initialization, device setup, and Kunlun memory profiling still execute through the MRO.
- `init_device` must call `super().init_device()` before assigning `self.model_runner`; the upstream worker initializes device and runtime state that the Kunlun model runners depend on.

## Omni-Specific Comment Markers

Omni-specific logic may be marked with comment blocks:

```python
# -------------------------------------- Omni-new -------------------------------------------------
# ... omni-specific code ...
# -------------------------------------- Omni-new -------------------------------------------------
```

Or simpler variations:

```python
#  -------------------------------------- Omni-new -------------------------------------------------
#  ------------------------------------------------------------------------------------------------
```

Important:

- Always preserve existing `Omni-new` markers when modifying code.
- Always grep GPU implementations for `Omni-new`; reference documents may not be up to date.
- The current Kunlun files may not contain all markers even when they contain omni-specific behavior. Also compare class inheritance, method bodies, comments, and calls to omni helpers.
- When discovering new omni-specific code that is not documented in references, update the reference files if the repo has them.

## Key Methods and Classes Requiring Attention

### OmniKunlunWorkerBase (`base.py`)

| Method | Description | Omni/Kunlun-specific Logic |
|--------|-------------|----------------------------|
| `load_model` | Load model through memory pool context | Uses `_maybe_get_memory_pool_context("weights")`, synchronizes `current_omni_platform`, runs GC |
| `__init__` | Worker initialization | Creates omni profiler for stage/rank-aware profiling |
| `profile` | Start/stop profiling | Sets trace filename for `OmniTorchProfilerWrapper` |
| `determine_available_memory` | Memory profiling | Uses process-scoped profiling and `current_omni_platform` memory accounting |
| `_maybe_get_memory_pool_context` | Sleep-mode memory pool | Uses `CuMemAllocator` because Kunlun XPU currently follows CUDA-compatible allocator paths in vllm-xpu |
| `sleep` / `wake_up` | Worker memory control | Frees/wakes memory via allocator and `current_omni_platform` |
| `handle_sleep_task` / `handle_wake_task` | Runtime sleep/wake commands | Emits `OmniACK`, handles distributed barriers, reports per-stage metadata |

### OmniKunlunModelRunner (`kunlun_model_runner.py`)

| Method | Description | Omni/Kunlun-specific Logic |
|--------|-------------|----------------------------|
| `load_model` | Model loading | Calls `XPUModelRunner.load_model` directly to preserve Kunlun MRO behavior while continuing into `OmniGPUModelRunner` |
| `_dummy_run` | Warmup/profiling run | Based on `XPUModelRunner._dummy_run`, preserves Kunlun padding, MoE router patches, spec-decode metadata, embed warmup, and omni talker/multimodal behavior |
| `_model_forward` if present/upstream changed | Main forward wrapper | Keep Kunlun backend execution structure while preserving omni model kwargs, multimodal output wrapping, and talker MTP logic |
| `_talker_mtp_forward` if present/upstream changed | Talker MTP forward | Use Kunlun-compatible forward context; do not introduce Ascend-specific context APIs |

### KunlunARModelRunner (`kunlun_ar_model_runner.py`)

| Method | Description | Omni/Kunlun-specific Logic |
|--------|-------------|----------------------------|
| Class MRO | AR model runner composition | `KunlunARModelRunner(OmniKunlunModelRunner, GPUARModelRunner)` keeps Kunlun base behavior and GPU AR omni execute/sample behavior |
| `_init_device_properties` | Device property setup | Sets `num_sms = None` because Kunlun does not expose GPU SM count the same way |
| `_sync_device` | Synchronization hook | Uses `current_omni_platform.synchronize()` |
| `execute_model` if overridden | Main AR inference entry | Preserve KV transfer, request-state handling, multimodal extraction, and `OmniModelRunnerOutput` |
| `sample_tokens` if overridden | Token sampling | Preserve hidden state extraction, multimodal outputs, and omni output packaging |

### KunlunGenerationModelRunner (`kunlun_generation_model_runner.py`)

| Method | Description | Omni/Kunlun-specific Logic |
|--------|-------------|----------------------------|
| Class MRO | Generation runner composition | `KunlunGenerationModelRunner(OmniKunlunModelRunner, GPUGenerationModelRunner)` keeps Kunlun base behavior and GPU generation omni behavior |
| `_init_device_properties` | Device property setup | Sets `num_sms = None` |
| `_sync_device` | Synchronization hook | Uses `current_omni_platform.synchronize()` |
| `_update_request_states` if overridden | Async generation state | Preserve async chunk behavior |
| `execute_model` if overridden | Generation forward | Preserve async chunk, `seq_token_counts`, generation kwargs, and `_run_generation_model` behavior |
| `sample_tokens` if overridden | Output processing | Preserve multimodal output packaging to `OmniModelRunnerOutput` |

### KunlunARWorker and KunlunGenerationWorker

| File | Description | Critical Behavior |
|------|-------------|-------------------|
| `kunlun_ar_worker.py` | AR worker | `init_device` must instantiate `KunlunARModelRunner(self.vllm_config, self.device)` after `super().init_device()` |
| `kunlun_generation_worker.py` | Generation worker | `init_device` must instantiate `KunlunGenerationModelRunner(self.vllm_config, self.device)` after `super().init_device()` |

## Upgrade Workflow

### Step 1: Preparation

1. Identify the target versions:
   - vllm-omni branch and last release/tag.
   - Local latest `vllm-xpu` code. Prefer the local installed/source checkout over guessed paths.
   - Any Kunlun-specific patches already present in this repo.

2. Locate current Kunlun backend definitions:

   ```bash
   grep -R "class XPUModelRunner\|class XPUWorker" <vllm-xpu-source>
   grep -R "vllm_xpu" vllm_omni/platforms/kunlun vllm_omni/platforms/__init__.py setup.py
   ```

3. Check recent GPU-side omni changes:

   ```bash
   git log --oneline -- vllm_omni/worker/
   git diff <from-tag>..<to-tag> -- vllm_omni/worker/gpu_model_runner.py
   git diff <from-tag>..<to-tag> -- vllm_omni/worker/gpu_ar_model_runner.py
   git diff <from-tag>..<to-tag> -- vllm_omni/worker/gpu_generation_model_runner.py
   ```

### Step 2: Analyze Omni-Specific Logic

For each Kunlun model runner and worker file:

1. Extract existing omni markers and omni helper calls:

   ```bash
   grep -n "Omni-new\|Omni\|omni\|talker\|multimodal\|extract_multimodal\|OmniModelRunnerOutput" \
     vllm_omni/platforms/kunlun/worker/*.py
   ```

2. Extract GPU-side authoritative omni logic:

   ```bash
   grep -n "Omni-new\|talker\|multimodal\|extract_multimodal\|OmniModelRunnerOutput" \
     vllm_omni/worker/gpu_model_runner.py \
     vllm_omni/worker/gpu_ar_model_runner.py \
     vllm_omni/worker/gpu_generation_model_runner.py
   ```

3. Document each relevant block:
   - Which class and method it belongs to.
   - Whether it is backend-independent omni behavior or GPU-specific implementation detail.
   - Whether Kunlun already inherits it through MRO or needs an explicit override.

### Step 3: Update `OmniKunlunWorkerBase`

1. Compare `base.py` with latest `vllm_xpu.v1.worker.xpu_worker.XPUWorker`.
2. Preserve Kunlun worker initialization and device behavior from `XPUWorker`.
3. Preserve omni worker behavior:
   - memory pool context around weight loading;
   - `current_omni_platform.synchronize()` calls;
   - omni profiler setup;
   - process-scoped memory profiling;
   - sleep/wake task handling and `OmniACK` emission.
4. Keep `CuMemAllocator` usage only because local Kunlun/vllm-xpu currently uses CUDA-compatible allocator paths. If upstream vllm-xpu introduces a Kunlun-specific allocator, migrate to that allocator.
5. Avoid replacing `current_omni_platform` calls with raw `torch.cuda` unless there is no platform abstraction available.

### Step 4: Update `OmniKunlunModelRunner`

1. Read the latest local `XPUModelRunner.load_model` and `_dummy_run`.
2. Keep `XPUModelRunner` first in the class MRO.
3. Keep direct `XPUModelRunner.load_model(self, *args, **kwargs)` unless upstream MRO changes require a different call pattern.
4. Update `_dummy_run` from latest `XPUModelRunner._dummy_run`, then reinsert omni behavior from `OmniGPUModelRunner._dummy_run`:
   - `has_preprocess` input path;
   - attention metadata extensions;
   - `talker_mtp` warmup;
   - `extract_multimodal_outputs`;
   - multimodal dummy kwargs and embed warmup.
5. Preserve Kunlun-specific behavior:
   - query-start padding and batch descriptors;
   - `maybe_create_ubatch_slices` handling;
   - XPU MoE router patches from `vllm_xpu.model_executor.layers.fused_moe`;
   - speculative decode metadata and drafter dummy runs;
   - `torch.cuda.synchronize()` only where vllm-xpu still exposes Kunlun through CUDA-compatible APIs.
6. If upstream adds `_model_forward`, graph capture, or forward-context changes, copy the Kunlun backend structure first and reinsert omni model kwargs, `OmniOutput`, multimodal extraction, and talker logic.

### Step 5: Update `KunlunARModelRunner`

1. Compare with latest `GPUARModelRunner` for omni-specific AR behavior.
2. If the current MRO still provides all required behavior, keep the file minimal.
3. Add explicit overrides only when Kunlun requires backend-specific handling or upstream GPU logic stops composing correctly through MRO.
4. Preserve these omni behaviors if overriding:
   - `OmniKVTransferManager` and KV transfer handling;
   - `_resolve_global_request_id`;
   - request-state updates;
   - hidden-state extraction;
   - multimodal output extraction;
   - `OmniModelRunnerOutput` packaging;
   - `_sync_device` using `current_omni_platform.synchronize()`.

### Step 6: Update `KunlunGenerationModelRunner`

1. Compare with latest `GPUGenerationModelRunner` for generation-specific omni behavior.
2. Keep the file minimal if MRO composition remains sufficient.
3. Add explicit overrides only when Kunlun backend execution requires them.
4. Preserve these omni behaviors if overriding:
   - async chunk request-state handling;
   - `seq_token_counts` injection;
   - `_run_generation_model` semantics;
   - generation kwargs initialization;
   - multimodal output extraction;
   - `OmniModelRunnerOutput` packaging;
   - `_sync_device` using `current_omni_platform.synchronize()`.

### Step 7: Update Kunlun Workers

1. Compare `KunlunARWorker` and `KunlunGenerationWorker` with latest `XPUWorker` initialization expectations.
2. Preserve `OmniWorkerMixin` first in the worker MRO.
3. Preserve `super().init_device()` before assigning the model runner.
4. Ensure workers instantiate the correct Kunlun model runner classes:

   ```python
   self.model_runner = KunlunARModelRunner(self.vllm_config, self.device)
   self.model_runner = KunlunGenerationModelRunner(self.vllm_config, self.device)
   ```

5. Update `__init__.py` exports when adding or renaming worker classes.

### Step 8: Update Imports

Check and update imports at the top of each file.

Common Kunlun imports currently used:

```python
from vllm_xpu.v1.worker.xpu_runner import XPUModelRunner
from vllm_xpu.v1.worker.xpu_worker import XPUWorker as KunlunWorker
from vllm_xpu.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias
from vllm_xpu.model_executor.layers.fused_moe.router.grouped_topk_router import fused_grouped_topk
```

Common omni imports currently used:

```python
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin
from vllm_omni.diffusion.data import OmniACK, OmniSleepTask, OmniWakeTask
```

Do not guess new `vllm_xpu` paths. Verify the local backend before changing imports.

### Step 9: Sync GPU-Side Omni Changes

1. Check recent GPU worker changes:

   ```bash
   git diff <from-tag>..<to-tag> -- vllm_omni/worker/gpu_model_runner.py
   git diff <from-tag>..<to-tag> -- vllm_omni/worker/gpu_ar_model_runner.py
   git diff <from-tag>..<to-tag> -- vllm_omni/worker/gpu_generation_model_runner.py
   git diff <from-tag>..<to-tag> -- vllm_omni/worker/mixins.py
   ```

2. Identify new omni features that need to be inherited or explicitly ported to Kunlun.
3. Prefer preserving behavior through MRO when it is correct and readable.
4. Use explicit Kunlun overrides when backend-specific execution order, context, allocator behavior, or device sync requires it.

### Step 10: Validation

1. Run syntax checks:

   ```bash
   python -m py_compile vllm_omni/platforms/kunlun/worker/base.py
   python -m py_compile vllm_omni/platforms/kunlun/worker/kunlun_model_runner.py
   python -m py_compile vllm_omni/platforms/kunlun/worker/kunlun_ar_model_runner.py
   python -m py_compile vllm_omni/platforms/kunlun/worker/kunlun_generation_model_runner.py
   python -m py_compile vllm_omni/platforms/kunlun/worker/kunlun_ar_worker.py
   python -m py_compile vllm_omni/platforms/kunlun/worker/kunlun_generation_worker.py
   ```

2. Run import test:

   ```bash
   python -c "from vllm_omni.platforms.kunlun.worker import *"
   ```

3. If Kunlun hardware/runtime is available, run a minimal serve or profile run:

   ```bash
   vllm serve <model-path> --trust-remote-code
   ```

4. If hardware is not available, report that runtime validation was not performed.

## Common Pitfalls

### 1. Treating Kunlun as Plain CUDA

Kunlun currently uses CUDA-compatible torch APIs in some vllm-xpu paths, but this is an implementation detail. Prefer `current_omni_platform` abstractions for synchronization, memory queries, and cache control when vllm-omni provides them.

### 2. Breaking MRO Composition

The Kunlun runner intentionally composes `XPUModelRunner` with omni GPU runner classes. Before adding large overrides, check whether the desired omni behavior is already inherited. Avoid duplicating GPU AR/generation methods unless Kunlun needs backend-specific changes.

### 3. Losing XPU Backend Fixes

When updating `_dummy_run`, preserve XPU-specific fixes such as MoE router patches, batch padding behavior, speculative decode metadata, and embed warmup. Copying directly from GPU code can drop Kunlun backend fixes.

### 4. Losing Omni Behavior

When copying from `XPUModelRunner`, reinsert omni behavior such as talker MTP warmup, multimodal dummy inputs, `extract_multimodal_outputs`, model kwargs injection, and `OmniModelRunnerOutput` packaging.

### 5. Raw `torch.cuda` Calls

Some raw `torch.cuda` calls may remain because vllm-xpu exposes CUDA-compatible APIs. Audit each occurrence. Keep it only when it matches local vllm-xpu behavior; otherwise prefer `current_omni_platform`.

### 6. Worker Initialization Order

Do not instantiate `KunlunARModelRunner` or `KunlunGenerationModelRunner` before `super().init_device()`. The base worker sets device/runtime state needed by the runner.

## Backend API Mapping

| Source Pattern | Meaning | Kunlun Action |
|----------------|---------|---------------|
| `CUDAGraphWrapper` | CUDA graph capture | Use only if vllm-xpu explicitly relies on this compatibility path |
| `set_forward_context` | vLLM forward context | Current Kunlun runner uses this; update only if vllm-xpu changes |
| `torch.cuda.*` | CUDA-compatible runtime call | Audit and prefer `current_omni_platform` where possible |
| `XPUModelRunner` | Kunlun backend runner | Primary model-runner baseline |
| `XPUWorker` | Kunlun backend worker | Primary worker baseline |

## Checklist Before Commit

- [ ] Latest local `vllm_xpu` runner/worker code was checked.
- [ ] GPU-side omni logic was checked, not only old reference docs.
- [ ] `OmniKunlunModelRunner` keeps `XPUModelRunner` first in the MRO.
- [ ] AR and generation runners still inherit the correct GPU omni runner classes.
- [ ] Worker MRO keeps `OmniWorkerMixin` and `OmniKunlunWorkerBase` behavior.
- [ ] `current_omni_platform.synchronize()` is used for Kunlun sync hooks.
- [ ] Sleep/wake and profiling behavior in `base.py` is preserved.
- [ ] Raw `torch.cuda` calls were audited.
- [ ] No duplicate overrides were added when MRO inheritance was sufficient.
- [ ] `__init__.py` exports are updated if classes were renamed or added.
- [ ] `py_compile` passes for changed Kunlun worker files.
- [ ] Import test passes or the missing dependency/runtime reason is documented.

## Reference Files for Comparison

When upgrading, keep these files open for reference:

1. Kunlun target base worker: `vllm_omni/platforms/kunlun/worker/base.py`
2. Kunlun target base runner: `vllm_omni/platforms/kunlun/worker/kunlun_model_runner.py`
3. Kunlun AR runner: `vllm_omni/platforms/kunlun/worker/kunlun_ar_model_runner.py`
4. Kunlun generation runner: `vllm_omni/platforms/kunlun/worker/kunlun_generation_model_runner.py`
5. GPU base runner: `vllm_omni/worker/gpu_model_runner.py`
6. GPU AR runner: `vllm_omni/worker/gpu_ar_model_runner.py`
7. GPU generation runner: `vllm_omni/worker/gpu_generation_model_runner.py`
8. Local vllm-xpu runner: file containing `vllm_xpu.v1.worker.xpu_runner.XPUModelRunner`
9. Local vllm-xpu worker: file containing `vllm_xpu.v1.worker.xpu_worker.XPUWorker`
