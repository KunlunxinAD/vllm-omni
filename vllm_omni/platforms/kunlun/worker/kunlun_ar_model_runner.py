# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kunlun XPU autoregressive model runner for vLLM-Omni."""

from __future__ import annotations

from vllm_omni.platforms import current_omni_platform
from vllm_omni.platforms.kunlun.worker.kunlun_model_runner import OmniKunlunModelRunner
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


class KunlunARModelRunner(OmniKunlunModelRunner, GPUARModelRunner):
    """Kunlun AR runner for thinker/talker stages.

    The MRO keeps Kunlun/XPU load, forward, and dummy-run fixes from
    ``OmniKunlunModelRunner`` while using ``GPUARModelRunner`` for Omni's
    autoregressive execute/sample behavior.
    """

    def _init_device_properties(self):
        self.num_sms = None

    def _sync_device(self) -> None:
        current_omni_platform.synchronize()
