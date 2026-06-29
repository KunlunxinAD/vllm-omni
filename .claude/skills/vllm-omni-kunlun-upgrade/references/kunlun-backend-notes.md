# Kunlun Backend Notes

This document records Kunlun-specific backend assumptions for the vllm-omni Kunlun upgrade skill. Verify these assumptions against local `vllm_xpu` code during every upgrade.

## Current Integration Points

The current vllm-omni Kunlun stack imports these `vllm_xpu` components:

```python
from vllm_xpu.v1.worker.xpu_runner import XPUModelRunner
from vllm_xpu.v1.worker.xpu_worker import XPUWorker as KunlunWorker
from vllm_xpu.platforms.kunlun import XPU3Platform
```

Fused MoE router patches currently import:

```python
from vllm_xpu.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias
from vllm_xpu.model_executor.layers.fused_moe.router.grouped_topk_router import fused_grouped_topk
```

If any of these paths change upstream, update Kunlun worker/runner imports and this reference.

## Platform Detection

Kunlun platform detection appears in:

```text
setup.py
vllm_omni/platforms/__init__.py
vllm_omni/platforms/kunlun/platform.py
```

Before diagnosing import/runtime errors, check that `vllm_xpu.platforms.kunlun.XPU3Platform` is still available and that vllm-omni's platform probing still recognizes the environment.

## CUDA-Compatible APIs

Kunlun XPU currently uses some CUDA-compatible torch/vLLM APIs. Examples in current code include:

```python
from vllm.config import CUDAGraphMode
from vllm.device_allocator.cumem import CuMemAllocator
torch.cuda.synchronize()
torch.cuda.get_device_properties(self.device).total_memory
```

These calls are not automatically wrong in Kunlun code, but they are fragile assumptions. During upgrades:

- Prefer `current_omni_platform` for synchronization and memory operations.
- Keep CUDA-compatible APIs only when local `vllm_xpu` expects them.
- Look for upstream vllm-xpu native alternatives before editing.

## MRO Design

Kunlun model runners intentionally rely on multiple inheritance rather than copying all GPU omni methods.

```python
class OmniKunlunModelRunner(XPUModelRunner, OmniGPUModelRunner):
    ...

class KunlunARModelRunner(OmniKunlunModelRunner, GPUARModelRunner):
    ...

class KunlunGenerationModelRunner(OmniKunlunModelRunner, GPUGenerationModelRunner):
    ...
```

Why this matters:

- `XPUModelRunner` preserves backend-specific load, dummy-run, tracing, MoE, attention, and graph behavior.
- `OmniGPUModelRunner` preserves omni base behavior.
- `GPUARModelRunner` and `GPUGenerationModelRunner` preserve omni execute/sample behavior.

Before adding explicit overrides, inspect the MRO and check whether inherited behavior is already correct:

```python
python - <<'PY'
from vllm_omni.platforms.kunlun.worker.kunlun_model_runner import OmniKunlunModelRunner
from vllm_omni.platforms.kunlun.worker.kunlun_ar_model_runner import KunlunARModelRunner
from vllm_omni.platforms.kunlun.worker.kunlun_generation_model_runner import KunlunGenerationModelRunner
print(OmniKunlunModelRunner.__mro__)
print(KunlunARModelRunner.__mro__)
print(KunlunGenerationModelRunner.__mro__)
PY
```

## Worker Design

Kunlun workers use `OmniWorkerMixin` plus `OmniKunlunWorkerBase`:

```python
class KunlunARWorker(OmniWorkerMixin, OmniKunlunWorkerBase):
    def init_device(self):
        super().init_device()
        self.model_runner = KunlunARModelRunner(self.vllm_config, self.device)

class KunlunGenerationWorker(OmniWorkerMixin, OmniKunlunWorkerBase):
    def init_device(self):
        super().init_device()
        self.model_runner = KunlunGenerationModelRunner(self.vllm_config, self.device)
```

Keep `super().init_device()` before replacing `self.model_runner`; upstream worker initialization sets device/runtime state needed by the runner.

## Sleep/Wake Memory Control

`OmniKunlunWorkerBase` supports sleep/wake tasks using:

- `OmniSleepTask`
- `OmniWakeTask`
- `OmniACK`
- `CuMemAllocator`
- `current_omni_platform`

This is part of vllm-omni's runtime control plane. Do not remove it when syncing from `XPUWorker`.

Important behaviors:

- Rank 0 emits successful ACKs.
- Distributed workers participate in all-reduce/barrier when initialized.
- Level 2 sleep clears graph runners when present.
- Memory freed/residual metadata is reported in GiB.

## Profiler Behavior

Kunlun worker profiling is omni-specific:

```python
from vllm_omni.profiler import create_omni_profiler
from vllm_omni.profiler import OmniTorchProfilerWrapper
```

The worker name uses stage and rank:

```python
worker_name = f"stage{stage_id}_rank{self.rank}"
```

Preserve this to keep profiles separable in multi-stage deployments.

## Dummy Run Behavior

Kunlun `_dummy_run` is the highest-risk method during upgrades. It combines:

- latest XPU batch and padding behavior;
- vLLM forward context behavior;
- XPU MoE router patches;
- speculative decode dummy-run behavior;
- LoRA dummy-run behavior;
- multimodal dummy inputs;
- omni talker MTP warmup;
- omni multimodal output extraction;
- optional embed warmup.

When upstream `XPUModelRunner._dummy_run` changes, copy the upstream structure first, then reinsert omni behavior. Do not start from GPU or NPU `_dummy_run` as the base.

## Do Not Import Ascend APIs

These imports are NPU/Ascend-specific and should not appear in Kunlun code:

```python
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
```

If a future vllm-xpu compatibility layer intentionally references one of these, document why in code and update this note.

## Minimal Validation Commands

```bash
python -m py_compile vllm_omni/platforms/kunlun/worker/base.py
python -m py_compile vllm_omni/platforms/kunlun/worker/kunlun_model_runner.py
python -m py_compile vllm_omni/platforms/kunlun/worker/kunlun_ar_model_runner.py
python -m py_compile vllm_omni/platforms/kunlun/worker/kunlun_generation_model_runner.py
python -m py_compile vllm_omni/platforms/kunlun/worker/kunlun_ar_worker.py
python -m py_compile vllm_omni/platforms/kunlun/worker/kunlun_generation_worker.py
python -c "from vllm_omni.platforms.kunlun.worker import *"
```

If these fail because `vllm_xpu` or Kunlun runtime dependencies are unavailable, report that clearly rather than masking the failure.
