# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kunlun XPU generation worker for vLLM-Omni."""

from __future__ import annotations

from vllm_omni.platforms.kunlun.worker.base import OmniKunlunWorkerBase
from vllm_omni.platforms.kunlun.worker.kunlun_generation_model_runner import KunlunGenerationModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin


class KunlunGenerationWorker(OmniWorkerMixin, OmniKunlunWorkerBase):
    """Kunlun generation worker for code2wav stages in Omni models."""

    def init_device(self):
        super().init_device()
        self.model_runner = KunlunGenerationModelRunner(self.vllm_config, self.device)
