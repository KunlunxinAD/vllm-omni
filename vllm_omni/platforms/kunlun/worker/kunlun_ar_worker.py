# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kunlun XPU AR worker for vLLM-Omni."""

from __future__ import annotations

from vllm_omni.platforms.kunlun.worker.base import OmniKunlunWorkerBase
from vllm_omni.platforms.kunlun.worker.kunlun_ar_model_runner import KunlunARModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin


class KunlunARWorker(OmniWorkerMixin, OmniKunlunWorkerBase):
    """Kunlun AR worker for thinker/talker stages in Omni models."""

    def init_device(self):
        super().init_device()
        self.model_runner = KunlunARModelRunner(self.vllm_config, self.device)
