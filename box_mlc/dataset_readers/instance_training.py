from typing import (
    Dict,
    List,
    Any,
    Iterator,
    cast,
    Tuple,
    Iterable,
)
import sys
import logging
import json
import numpy as np
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
    target: ArrayField  #: types
    labels: MultiLabelField


@DatasetReader.register("box-target")
class BoxTargetReader(DatasetReader):
    def example_to_fields(
        self,
        x: List[float],
        y: List[str],
        target: List[float],
        idx: str,
        meta: Dict = None,
        **kwargs: Any,
    ) -> InstanceFields:
        if meta is None:
            meta = {}

        meta["x"] = x
        meta["labels"] = y
        meta["using_tc"] = False

        x_field = ArrayField(np.array(x))
        labels_field = MultiLabelField(y)
        target_field = ArrayField(np.array(target))

        return {"x": x_field, "labels": labels_field, "target": target_field}

    def text_to_instance(  # type:ignore
        self,
        x: List[float],
        y: List[str],
        target: List[float],
        idx: str,
        **kwargs: Any,
    ) -> Instance:
        meta_dict: Dict = {}
        main_fields = self.example_to_fields(
            x, y, target, idx, meta=meta_dict, **kwargs
        )

        return Instance(
            {**cast(dict, main_fields), "meta": MetadataField(meta_dict)}
        )

    def _read(self, file_path: str) -> Iterator[Instance]:
        """Reads a datafile to produce instances

        Args:
            file_path: glob pattern for files containing folds to read

        Yields:
            data instances

        """
        with open(file_path) as f:
            data = json.load(f)

        for ex in data:
            yield self.text_to_instance(**ex)
