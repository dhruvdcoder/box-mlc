"""Dataset reader for toy_data_1"""
from typing import (
    Dict,
    List,
    Any,
    Iterator,
    cast,
    Iterable,
)
import sys
import logging
import json
import numpy as np
import dill
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField, MultiLabelField
from allennlp.data.instance import Instance

if sys.version_info >= (3, 8):
    from typing import (
        TypedDict,
    )  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class InstanceFields(TypedDict):
    """Contents which form an instance"""

    x: ArrayField
    labels: MultiLabelField  #: types


@DatasetReader.register("toy-data-1")
class ToyData1Reader(DatasetReader):
    """
    Multi-label classification toy dataset 1. Each data point has x
    which is  of size (2,) and labels between A to I.
    """

    def __init__(
        self,
        truncate_ratio: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Arguments:
            truncate_ratio: float, between 0 and 1. Fraction of data to be truncated after reading.
            **kwargs: Parent class args.
                `Reference <https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_reader.py>`_

        """
        super().__init__(**kwargs)
        self.truncate_ratio = truncate_ratio

    def example_to_fields(
        self,
        x: List[float],
        labels: List[str],
        meta: Dict = None,
        **kwargs: Any,
    ) -> InstanceFields:
        """Converts a dictionary containing an example datapoint to fields that can
        be used to create an :class:`Instance`. If a meta dictionary is passed,
        then it also adds raw data in the meta dict.

        Args:
            x: data point in the 2D space
            labels: toy data true labels
            meta: None
            **kwargs: TODO

        Returns:
            Dictionary of fields with the following entries:
                x: contains the x.
                labels: contains the labels.

        """

        if meta is None:
            meta = {}

        meta["x"] = x
        meta["labels"] = labels
        meta["using_tc"] = False

        x_field = ArrayField(np.array(x))
        labels_field = MultiLabelField(labels)

        return {
            "x": x_field,
            "labels": labels_field,
        }

    def text_to_instance(  # type:ignore
        self,
        x: List[float],
        labels: List[str],
        **kwargs: Any,
    ) -> Instance:
        """Converts contents of a single raw example to :class:`Instance`.

        Args:
            x: data point in the 2D space
            labels: toy data true labels
            meta: None
            **kwargs: TODO

        Returns:
             :class:`Instance` of data

        """
        meta_dict: Dict = {}
        main_fields = self.example_to_fields(x, labels, meta=meta_dict)

        return Instance(
            {**cast(dict, main_fields), "meta": MetadataField(meta_dict)}
        )

    def _read(self, file_path: str) -> Iterator[Instance]:
        """Reads a datafile to produce instances

        Args:
            file_path: TODO

        Yields:
            data instances

        """
        with open(file_path) as f:
            data = json.load(f)

        read_len = int((1 - self.truncate_ratio) * len(data))

        for idx, example in enumerate(data):
            instance = self.text_to_instance(**example)

            if idx < read_len:
                yield instance

    def _instances_from_cache_file(
        self, cache_filename: str
    ) -> Iterable[Instance]:
        logger.info(f"Reading instances from cache at {cache_filename}")
        with open(cache_filename, "rb") as cache_file:
            instances = dill.load(cache_file)

            for instance in instances:
                yield instance

    def _instances_to_cache_file(
        self, cache_filename: str, instances: Iterable[Instance]
    ) -> None:
        logger.info(f"Writing instances to cache at {cache_filename}")
        with open(cache_filename, "wb") as cache_file:
            dill.dump(instances, cache_file)
