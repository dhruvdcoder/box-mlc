from typing import (
    Dict,
    List,
    Any,
    Iterator,
    cast,
    Tuple,
    Iterable,
)

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
import numpy as np
from .arff_reader import ARFFReader
from allennlp.data.fields import ArrayField, TextField, LabelField
from .arff_reader import InstanceFields as BaseInstanceFields
from allennlp.data.fields import ArrayField, MetadataField, MultiLabelField
import sys

if sys.version_info >= (3, 8):
    from typing import (
        TypedDict,
    )  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict


import logging

logger = logging.getLogger(__name__)


class InstanceFields(TypedDict):
    x: ArrayField
    idx: LabelField  # we do not want oov/unk tokens, as well as indexer embedder pipeline, hence use LabelField
    labels: MultiLabelField


@DatasetReader.register("arff-with-instance-id")
class ARFFWithInstanceIdReader(ARFFReader):
    def example_to_fields(
        self,
        x: List[float],
        labels: List[str],
        meta: Dict = None,
        **kwargs: Any,
    ) -> InstanceFields:
        base_fields: BaseInstanceFields = super().example_to_fields(
            x, labels, meta
        )
        base_fields = cast(InstanceFields, base_fields)  # just for mypy.
        # add idx
        base_fields["idx"] = LabelField(
            kwargs["idx"], label_namespace="instance_tags"
        )  # we want _tags for the indexer to skip adding ovv/unk tokens

        return base_fields
