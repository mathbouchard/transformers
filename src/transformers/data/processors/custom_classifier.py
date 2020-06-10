# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Custom classifier processor and helpers """

import logging
import os
from enum import Enum
from typing import List, Optional, Union

from ...file_utils import is_tf_available
from ...tokenization_utils import PreTrainedTokenizer
from .utils import DataProcessor, InputExample, InputFeatures


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def custom_classifier_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    separator: str,
    max_length: int,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        return _tf_custom_classifier_convert_examples_to_features(examples, tokenizer, separator, max_length=max_length, task=task)
    return _custom_classifier_convert_examples_to_features(
        examples, tokenizer, separator, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
    )


if is_tf_available():

    def _tf_custom_classifier_convert_examples_to_features(
        examples: tf.data.Dataset, tokenizer: PreTrainedTokenizer, separator: str, task=str, max_length: Optional[int] = None,
    ) -> tf.data.Dataset:
        """
        Returns:
            A ``tf.data.Dataset`` containing the task-specific features.

        """
        
        def get_example_from_tensor_dict(tensor_dict):
            return InputExample(
                tensor_dict["idx"].numpy(),
                tensor_dict["sentence1"].numpy().decode("utf-8"),
                tensor_dict["sentence2"].numpy().decode("utf-8"),
                str(tensor_dict["label"].numpy()),
            )
        processor = DataProcessor()
        examples = [processor.tfds_map(get_example_from_tensor_dict(example)) for example in examples]
        features = custom_classifier_convert_examples_to_features(examples, tokenizer, separator, max_length=max_length, task=task)

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

def _custom_classifier_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    separator: str,
    max_length: int,
    task=None,
    label_list=None,
    output_mode=None,
):
    
    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example["label"] is None:
            return None
        return label_map[example["label"]]
        
    labels = [label_from_example(example) for example in examples]

    def get_sentence_parts(example):
        text_to_classify_tokens = (example['text_to_classify']+" "+separator).split()
        if(len(text_to_classify_tokens) > max_length):
            text_to_classify_tokens = text_to_classify_tokens[:max_length-1]+[separator]
        prefix_context_tokens = (example["prefix_context"]+" "+separator).split()
        suffix_context_tokens = example["suffix_context"].split()

        remaining_length = max_length - len(text_to_classify_tokens)
        
        if len(suffix_context_tokens) < remaining_length//2:
            length_2 = len(suffix_context_tokens)
            length_1 = remaining_length-length_2
        elif len(prefix_context_tokens) < remaining_length//2:
            length_1 = len(prefix_context_tokens)
            length_2 = remaining_length-length_1
        else:
            length_1 = remaining_length//2
            length_2 = remaining_length//2
        return (' '.join(prefix_context_tokens[-length_1:]).strip(),
                ' '.join(text_to_classify_tokens+suffix_context_tokens[:length_2]).strip())
    
    batch_encoding = tokenizer.batch_encode_plus(
        [get_sentence_parts(example) for example in examples], max_length=max_length, pad_to_max_length=True
    )
    
    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example["guid"]))
        logger.info("features: %s" % features[i])

    return features
