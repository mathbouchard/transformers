# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from ...file_utils import (
    _BaseLazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_bridgeformer": ["BRIDGEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "BridgeformerConfig"],
    "tokenization_bridgeformer": ["BridgeformerTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_bridgeformer_fast"] = ["BridgeformerTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_bridgeformer"] = [
        "BRIDGEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BridgeformerForCausalLM",
        "BridgeformerForMaskedLM",
        "BridgeformerForMultipleChoice",
        "BridgeformerForQuestionAnswering",
        "BridgeformerForSequenceClassification",
        "BridgeformerForTokenClassification",
        "BridgeformerModel",
    ]

if is_tf_available():
    _import_structure["modeling_tf_bridgeformer"] = [
        "TF_BRIDGEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFBridgeformerForMaskedLM",
        "TFBridgeformerForMultipleChoice",
        "TFBridgeformerForQuestionAnswering",
        "TFBridgeformerForSequenceClassification",
        "TFBridgeformerForTokenClassification",
        "TFBridgeformerMainLayer",
        "TFBridgeformerModel",
        "TFBridgeformerPreTrainedModel",
    ]

if is_flax_available():
    _import_structure["modeling_flax_bridgeformer"] = ["FlaxBridgeformerModel"]


if TYPE_CHECKING:
    from .configuration_bridgeformer import BRIDGEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, BridgeformerConfig
    from .tokenization_bridgeformer import BridgeformerTokenizer

    if is_tokenizers_available():
        from .tokenization_bridgeformer_fast import BridgeformerTokenizerFast

    if is_torch_available():
        from .modeling_bridgeformer import (
            BRIDGEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            BridgeformerForCausalLM,
            BridgeformerForMaskedLM,
            BridgeformerForMultipleChoice,
            BridgeformerForQuestionAnswering,
            BridgeformerForSequenceClassification,
            BridgeformerForTokenClassification,
            BridgeformerModel,
        )

    if is_tf_available():
        from .modeling_tf_bridgeformer import (
            TF_BRIDGEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFBridgeformerForMaskedLM,
            TFBridgeformerForMultipleChoice,
            TFBridgeformerForQuestionAnswering,
            TFBridgeformerForSequenceClassification,
            TFBridgeformerForTokenClassification,
            TFBridgeformerMainLayer,
            TFBridgeformerModel,
            TFBridgeformerPreTrainedModel,
        )

    if is_flax_available():
        from .modeling_flax_bridgeformer import FlaxBridgeformerModel

else:
    import importlib
    import os
    import sys

    class _LazyModule(_BaseLazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)
