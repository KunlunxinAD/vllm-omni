# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kunlun XPU generation model runner for vLLM-Omni."""

from __future__ import annotations

from vllm_omni.platforms import current_omni_platform
from vllm_omni.platforms.kunlun.worker.kunlun_model_runner import OmniKunlunModelRunner
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner


class KunlunGenerationModelRunner(OmniKunlunModelRunner, GPUGenerationModelRunner):
    """Kunlun generation runner for non-autoregressive generation stages."""

    def _init_device_properties(self):
        self.num_sms = None

    def _sync_device(self) -> None:
        current_omni_platform.synchronize()
