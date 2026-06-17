# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Base Kunlun XPU worker class for vLLM-Omni."""

from __future__ import annotations

import gc
import os
import time
from contextlib import AbstractContextManager, nullcontext

import torch
from vllm.logger import init_logger
from vllm.utils.mem_utils import format_gib, memory_profiling
from vllm_xpu.v1.worker.xpu_worker import XPUWorker as KunlunWorker

from vllm_omni.diffusion.data import (
    OmniACK,
    OmniSleepTask,
    OmniWakeTask,
)
from vllm_omni.entrypoints.utils import detect_pid_host
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


class OmniKunlunWorkerBase(KunlunWorker):
    """Base Kunlun XPU worker for vLLM-Omni.

    Kunlun XPU uses CUDA-compatible torch APIs in vLLM-XPU, so this class
    mirrors the Omni GPU worker interfaces while inheriting Kunlun-specific
    device and model-runner initialization from KunlunWorker.
    """

    def load_model(self, *args, **kwargs):
        with self._maybe_get_memory_pool_context("weights"):
            res = super().load_model(*args, **kwargs)
            current_omni_platform.synchronize()
            gc.collect()
            return res

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        """Set trace filename before starting the profiler."""
        if self.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. For diffusion models, set --profiler-config via CLI. "
                "For omni models, add profiler_config to your stage config file."
            )
        if is_start:
            from vllm_omni.profiler import OmniTorchProfilerWrapper

            if isinstance(self.profiler, OmniTorchProfilerWrapper):
                stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
                filename = profile_prefix or f"stage{stage_id}_rank{self.rank}_{int(time.time())}"
                self.profiler.set_trace_filename(filename)
            self.profiler.start()
        else:
            self.profiler.stop()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Process-scoped device memory profiling for concurrent initialization."""
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            self.model_runner.profile_run()
            return kv_cache_memory_bytes

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
        logger.debug(
            "Profiling fallback (PID %d, GPU %d): requested=%s, profiled=%s, available=%s",
            os.getpid(),
            self.local_rank,
            format_gib(self.requested_memory),
            format_gib(profiled_usage),
            format_gib(self.available_kv_cache_memory_bytes),
        )
        logger.info_once(
            "Available KV cache memory: %s GiB (profiling fallback)",
            format_gib(self.available_kv_cache_memory_bytes),
            scope="local",
        )

        return int(self.available_kv_cache_memory_bytes)

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        v1_config_enabled = False
        if hasattr(self, "vllm_config"):
            model_cfg = getattr(self.vllm_config, "model_config", None)
            v1_config_enabled = getattr(model_cfg, "enable_sleep_mode", False)

        is_sleep_enabled = v1_config_enabled or getattr(self.cache_config, "enable_sleep_mode", False)
        if is_sleep_enabled:
            current_omni_platform.synchronize()
            gc.collect()
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            logger.info(f"[LLM Worker {self.rank}] Sleep Mode ENABLED. Activating CuMem pool for tag: {tag}")
            return allocator.use_memory_pool(tag=tag)

        logger.warning(f"[LLM Worker {self.rank}] Sleep Mode DISABLED.")
        return nullcontext()

    def sleep(self, level: int = 1) -> bool:
        """Put the worker to sleep."""
        from vllm.device_allocator.cumem import CuMemAllocator

        mem_before = current_omni_platform.get_current_memory_usage(self.device)
        offload_tags = ("weights",) if level == 1 else tuple()
        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=offload_tags)
        current_omni_platform.empty_cache()
        current_omni_platform.synchronize()
        mem_after = current_omni_platform.get_current_memory_usage(self.device)
        freed = max(0, mem_before - mem_after)
        remaining_gb = mem_after / 1024**3
        logger.info(
            f"[LLM Worker {self.rank}] Level {level} Sleep: Freed "
            f"{freed / 1024**3:.2f} GiB. {remaining_gb:.2f}GiB memory "
            "is still in use."
        )
        return True

    def wake_up(self, tags: list[str] | None = None) -> bool:
        """Reload physical video memory."""
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)
        current_omni_platform.synchronize()
        logger.info(f"[LLM Worker {self.rank}] Wake-up complete.")
        return True

    def handle_sleep_task(self, task: OmniSleepTask) -> OmniACK:
        """Handle deterministic Sleep command from the main process."""
        try:
            if isinstance(task, dict):
                task = OmniSleepTask(**task)
            logger.info(f"[LLM Worker {self.rank}] Handshake Received: Task {task.task_id}, Level {task.level}")
            if task.level == 2:
                if hasattr(self.model_runner, "graph_runners"):
                    self.model_runner.graph_runners.clear()
                    logger.info(f"[LLM Worker {self.rank}] LLM CUDA Graphs cleared.")
            mem_before = current_omni_platform.get_free_memory(self.device)
            self.sleep(level=task.level)
            mem_after = current_omni_platform.get_free_memory(self.device)
            rank_freed = max(0, mem_after - mem_before)
            if torch.distributed.is_initialized():
                t_freed = torch.tensor([float(rank_freed)], device=self.device)
                torch.distributed.all_reduce(t_freed)
                total_freed = int(t_freed.item())
                torch.distributed.barrier()
            else:
                total_freed = rank_freed
            if self.rank != 0:
                return None
            current_stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
            try:
                total_mem = current_omni_platform.get_device_total_memory()
            except (NotImplementedError, AttributeError):
                total_mem = torch.cuda.get_device_properties(self.device).total_memory
            residual_gib = (total_mem - mem_after) / 1024**3
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
            logger.info(f"[LLM Worker {self.rank}] ACK emitted for Task {task.task_id}")
            return ack
        except Exception as e:
            logger.error(f"[LLM Worker {self.rank}] Sleep Task Failed: {e}", exc_info=True)
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception:
                    pass
            return OmniACK(task_id=task.task_id, status="ERROR", error_msg=str(e))

    def handle_wake_task(self, task: OmniWakeTask) -> OmniACK:
        """Handle deterministic Wakeup command from the main process."""
        try:
            if isinstance(task, dict):
                task = OmniWakeTask(**task)
            self.wake_up(tags=task.tags)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            gc.collect()
            current_omni_platform.synchronize()
            free_now = current_omni_platform.get_free_memory(self.device)
            try:
                total_mem = current_omni_platform.get_device_total_memory()
            except (NotImplementedError, AttributeError):
                total_mem = torch.cuda.get_device_properties(self.device).total_memory
            current_used_gib = (total_mem - free_now) / 1024**3
            if self.rank != 0:
                return None
            current_stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
            ack = OmniACK(
                task_id=task.task_id,
                status="SUCCESS",
                stage_id=current_stage_id,
                rank=self.rank,
                metadata={"state": "WARM", "current_vram_gib": f"{current_used_gib:.2f}"},
            )
            if hasattr(self, "result_mq") and self.result_mq:
                self.result_mq.put(ack)
            logger.info(f"[LLM Worker {self.rank}] Wake-up ACK emitted.")
            return ack
        except Exception as e:
            logger.error(f"[LLM Worker {self.rank}] Wake-up Failed: {e}", exc_info=True)
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception:
                    pass
            tid = task.task_id if hasattr(task, "task_id") else "unknown"
            return OmniACK(task_id=tid, status="ERROR", error_msg=str(e))
