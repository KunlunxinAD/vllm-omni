# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum
from vllm_xpu.platforms.kunlun import XPU3Platform as KunlunPlatformBase

logger = init_logger(__name__)


def _register_kunlun_cumem_alias() -> None:
    """Route vLLM CuMem imports to the XPU implementation on Kunlun."""
    import vllm.device_allocator as vllm_device_allocator
    import vllm_xpu.device_allocator.cumem as xpu_cumem

    sys.modules["vllm.device_allocator.cumem"] = xpu_cumem
    vllm_device_allocator.cumem = xpu_cumem


_register_kunlun_cumem_alias()


class KunlunOmniPlatform(OmniPlatform, KunlunPlatformBase):
    """Kunlun XPU implementation of OmniPlatform.

    Kunlun's vLLM platform exposes the device through CUDA-compatible torch APIs,
    so this platform keeps CUDA device semantics while adding Omni-specific hooks.
    """

    _omni_enum = OmniPlatformEnum.KUNLUN

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.platforms.kunlun.worker.kunlun_ar_worker.KunlunARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.platforms.kunlun.worker.kunlun_generation_worker.KunlunGenerationWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/model_executor/stage_configs"

    @classmethod
    def has_flash_attn_package(cls) -> bool:
        from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_func, flash_attn_varlen_func
        return flash_attn_func is not None or flash_attn_varlen_func is not None

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        if selected_backend is None:
            logger.debug("Defaulting Kunlun diffusion attention backend to SDPA")
            return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

        backend_upper = selected_backend.upper()
        backend = DiffusionAttentionBackendEnum[backend_upper]
        logger.debug("Using Kunlun diffusion attention backend '%s'", backend_upper)
        return backend.get_path()

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return False

    @classmethod
    def supports_float64(cls) -> bool:
        return False

    @classmethod
    def is_out_of_tree(cls) -> bool:
        return True

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank is None:
            return torch.device("cuda")
        return torch.device("cuda", local_rank)

    @classmethod
    def get_device_count(cls) -> int:
        return torch.accelerator.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        return torch.version.cuda

    @classmethod
    def synchronize(cls) -> None:
        torch.accelerator.synchronize()

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.cuda.mem_get_info(device)
        return free
