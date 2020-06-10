import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union, Dict, Callable

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from ...tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_xlm_roberta import XLMRobertaTokenizer
from ..processors.custom_classifier import custom_classifier_convert_examples_to_features
from ..processors.utils import InputFeatures


logger = logging.getLogger(__name__)


@dataclass
class CustomClassifierDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: custom"})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the data files for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class CustomClassifierDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: CustomClassifierDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]
    get_data: Callable
    get_labels: Callable
    custom_info: dict
    seperator: str


    def __init__(
        self,
        args: CustomClassifierDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        get_data: Callable,
        get_labels: Callable,
        custom_info: dict,
        separator: str,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None, 
    ):
        self.args = args
        self.output_mode = "classification"
        self.get_data = get_data
        self.get_labels = get_labels
        self.custom_info = custom_info
        self.separator = seperator
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        
        self.label_list = get_labels(self.custom_info)

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = get_data(args, custom_info)
                elif mode == Split.test:
                    examples = get_data(args, custom_info)
                else:
                    examples = get_data(args, custom_info)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = custom_classifier_convert_examples_to_features(
                    examples,
                    tokenizer,
                    separator,
                    max_length=args.max_seq_length,
                    label_list=self.label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list
