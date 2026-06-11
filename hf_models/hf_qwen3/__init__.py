# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .configuration_qwen3 import Qwen3Config
from .modeling_qwen3 import Qwen3Model, Qwen3ForCausalLM
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

try:
    from liger_kernel.transformers.model.qwen3 import lce_forward as qwen3_lce_forward
except ModuleNotFoundError:
    qwen3_lce_forward = None

AutoConfig.register("qwen3", Qwen3Config, exist_ok=True)
AutoModel.register(Qwen3Config, Qwen3Model, exist_ok=True)
AutoModelForCausalLM.register(Qwen3Config, Qwen3ForCausalLM, exist_ok=True)

_qwen3_forward = Qwen3ForCausalLM.forward


def _qwen3_lce_forward_with_ttt_aux(self, *args, **kwargs):
    if not self.lm_head.weight.is_cuda:
        return _qwen3_forward(self, *args, **kwargs)
    out = qwen3_lce_forward(self, *args, **kwargs)
    aux = getattr(self.model, "_last_ttt_aux_loss", None)
    if hasattr(out, "__dict__"):
        out.ttt_aux_loss = aux
    return out


if qwen3_lce_forward is not None:
    Qwen3ForCausalLM.forward = _qwen3_lce_forward_with_ttt_aux
