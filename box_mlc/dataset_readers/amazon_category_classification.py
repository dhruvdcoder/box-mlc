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

logger = logging.getLogger(__name__)


class InstanceFields(TypedDict):
    """Contents which form an instance"""

    sentences: ListField  #: it is actually ListField[TextField], one TextField instance per sentence
    mentions: ListField  #: again ListField[TextField]
    labels: MultiLabelField  #: types


@DatasetReader.register("amazon-category-classification")
class AmazonCategoryReader(DatasetReader):
    """
    Multi-label classification `dataset <http://manikvarma.org/downloads/XC/XMLRepository.html>`_.

    The output of this DatasetReader follows :class:`MultiInstanceEntityTyping`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        use_transitive_closure: bool = False,
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
        super().__init__(**kwargs)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self._use_transitive_closure = use_transitive_closure

    def example_to_fields(
        self,
        title: str,
        content: str,
        labels: List[str],
        uid: str,
        target_ind: List[int],
        target_rel: List[float],
        meta: Dict = None,
        **kwargs: Any,
    ) -> InstanceFields:
        """Converts a dictionary containing an example datapoint to fields that can be used
        to create an :class:`Instance`. If a meta dictionary is passed, then it also adds raw data in
        the meta dict.

        Args:
            title: Title of the item
            content: Contents of the category
            labels: Labels/Categories
            uid: unique id
            target_ind: original idx label mapping provided
            target_rel: target relevancy score
            meta: None
            **kwargs: TODO

        Returns:
            Dictionary of fields with the following entries:
                sentence: contains the content.
                mention: contains the title.

        """

        if meta is None:
            meta = {}

        meta["title"] = title
        meta["content"] = content
        meta["labels"] = labels
        meta["uid"] = uid
        meta["target_rel"] = target_rel
        meta["using_tc"] = False

        sentence_fields = ListField(
            [
                TextField(
                    self._tokenizer.tokenize(content),
                    token_indexers=self._token_indexers,
                )
            ]
        )
        mention_fields = ListField(
            [
                TextField(
                    self._tokenizer.tokenize(title),
                    token_indexers=self._token_indexers,
                )
            ]
        )
        labels_ = MultiLabelField(labels)

        return {
            "sentences": sentence_fields,
            "mentions": mention_fields,
            "labels": labels_,
        }

    def text_to_instance(  # type:ignore
        self,
        title: str,
        content: str,
        labels: List[str],
        uid: str,
        target_ind: List[int],
        target_rel: List[float],
        **kwargs: Any,
    ) -> Instance:
        """Converts contents of a single raw example to :class:`Instance`.

        Args:
            title: Title of the item
            content: Contents of the category
            labels: Labels/Categories
            uid: unique id
            target_ind: original idx label mapping provided
            target_rel: target relevancy score
            **kwargs: TODO

        Returns:
             :class:`Instance` of data

        """
        meta_dict: Dict = {}
        main_fields = self.example_to_fields(
            title, content, labels, uid, target_ind, target_rel, meta=meta_dict
        )

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

        for example in data:
            instance = self.text_to_instance(**example)

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
