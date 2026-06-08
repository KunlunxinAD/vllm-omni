# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum
from vllm_xpu.platforms.kunlun import XPU3Platform as KunlunPlatformBase

logger = init_logger(__name__)


class KunlunOmniPlatform(OmniPlatform, KunlunPlatformBase):
    """Kunlun XPU implementation of OmniPlatform.

    Kunlun's vLLM platform exposes the device through CUDA-compatible torch APIs,
    so this platform keeps CUDA device semantics while adding Omni-specific hooks.
    """

    _omni_enum = OmniPlatformEnum.KUNLUN

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_ar_worker.GPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/model_executor/stage_configs"

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.debug("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        logger.debug("Defaulting to diffusion attention backend SDPA")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

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
