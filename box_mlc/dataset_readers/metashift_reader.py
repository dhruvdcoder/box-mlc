from typing import (
    Dict,
    List,
    Union,
    Any,
    Iterator,
    Tuple,
    cast,
    Optional,
    Iterable,
)
import sys
import itertools

from tensorflow import Tensor

if sys.version_info >= (3, 8):
    from typing import (
        TypedDict,
    )  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict, Literal, overload

import logging
import json
import dill
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    TextField,
    Field,
    ArrayField,
    ListField,
    MetadataField,
    MultiLabelField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers import Token
from .common import JSONTransform
import glob
from datasets import load_from_disk

logger = logging.getLogger(__name__)


class InstanceFields(TypedDict):
    """Contents which form an instance"""

    image_id: str
    image: Tensor
    label: Field
    context: str


@DatasetReader.register("metashift")
class MetashiftReader(DatasetReader):
    """
    Multi-label classification `dataset <https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html>`_.

    The output of this DatasetReader follows :class:`MultiInstanceEntityTyping`.
    """

    def __init__(
        self,
        test: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Arguments:
            tokenizer: The tokenizer to be used.
            token_indexers: The token_indexers to be used--one per embedder. Will usually be only one.
            use_transitive_closure: use types_extra
            **kwargs: Parent class args.
                `Reference <https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_reader.py>`_

        """
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self.test = test
        # self._use_transitive_closure = use_transitive_closure

    def text_to_instance(  # type:ignore
        self,
        image_id: str,
        image: Tensor,
        label: Field,
        context: str,
        **kwargs: Any
    ) -> Instance:
        """Converts contents of a single raw example to :class:`Instance`.

        Args:
            text: One line summary of article,
            title: Title of the article
            labels:list of labels,
            general_descriptors: Extra descriptors,
            label_path: List of taxonomies,
            xml_path: path to xml file,
            taxonomy: Taxonomy extracted form xml file
            **kwargs: Any

        Returns:
             :class:`Instance` of data

        """
        meta_dict: Dict = {}
        # main_fields = self.example_to_fields(
        #     type, concepts, title, main_body, meta=meta_dict
        # )

        instance = {
            "image_id": image_id,
            "image": image,
            "label": label,
            "context": context
        }

        return Instance(
            {**instance, "meta": MetadataField(meta_dict)}
        )

    def _read(self, file_path: str) -> Iterator[Instance]:
        """Reads a datafile to produce instances

        Args:
            file_path: TODO

        Yields:
            data instances

        """
        dataset = load_from_disk(file_path)
        for file_count, file_ in self.shard_iterable(enumerate(glob.glob(file_path))):
            logger.info(f"Reading {file_}")
            with open(file_, mode='r') as f:
                line = f.readline()
                example = json.loads(line)
                instance = self.text_to_instance(**example)
                f.close()
                yield instance
            if self.test and file_count > 20:
                break

    # def apply_token_indexers(self, instance: Instance) -> None:
    #     instance["text"].token_indexers = self._token_indexers
