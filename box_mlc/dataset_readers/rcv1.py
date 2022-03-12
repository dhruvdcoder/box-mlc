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
from wcmatch import glob

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
import allennlp

allennlp_major_version = int(allennlp.__version__.split(".")[0])
logger = logging.getLogger(__name__)


class InstanceFields(TypedDict):
    """Contents which form an instance"""

    sentences: ListField  #: it is actually ListField[TextField], one TextField instance per sentence
    mentions: ListField  #: again ListField[TextField]
    labels: MultiLabelField  #: types


if allennlp_major_version < 2:
    pass

elif allennlp_major_version >= 2:

    @DatasetReader.register("rcv1")
    class RCV1(DatasetReader):
        """"""

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
            super().__init__(
                manual_distributed_sharding=True,
                manual_multiprocess_sharding=True,
                **kwargs,
            )
            self._tokenizer = tokenizer
            self._token_indexers = token_indexers
            self._use_transitive_closure = use_transitive_closure

        def example_to_fields(
            self,
            headline: str,
            text: str,
            labels: List[str],
            idx: str,
            meta: Dict = None,
            **kwargs: Any,
        ) -> InstanceFields:
            """Converts a dictionary containing an example datapoint to fields that can be used
            to create an :class:`Instance`. If a meta dictionary is passed, then it also adds raw data in
            the meta dict.

            Returns:
                Dictionary of fields with the following entries:
                    sentence: contains the body.
                    mention: contains the title.

            """

            if meta is None:
                meta = {}

            #meta["headline"] = headline
            #meta["text"] = text
            meta["labels"] = labels
            meta["idx"] = idx

            sentence_fields = ListField(
                [
                    TextField(
                        self._tokenizer.tokenize(text),
                    )
                ]
            )
            mention_fields = ListField(
                [
                    TextField(
                        self._tokenizer.tokenize(headline),
                    )
                ]
            )

            # if self._use_transitive_closure:
            #   types += types_extra
            #    meta["using_tc"] = True

            return {
                "sentences": sentence_fields,
                "mentions": mention_fields,
                "labels": MultiLabelField(labels),
            }

        def text_to_instance(  # type:ignore
            self,
            headline: str,
            text: str,
            labels: List[str],
            idx: str,
            **kwargs: Any,
        ) -> Instance:
            """Converts contents of a single raw example to :class:`Instance`.

            Returns:
                 :class:`Instance` of data

            """
            meta_dict: Dict = {}
            main_fields = self.example_to_fields(
                headline or "", text or "", labels, idx, meta=meta_dict
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
            files = list(glob.glob(file_path, flags=glob.EXTGLOB))
            if not files:
                raise RuntimeError(f"{file_path} glob did not match any files")
            for file_ in files:
                with open(file_) as f:
                    for line in self.shard_iterable(f):
                        example = json.loads(line)
                        instance = self.text_to_instance(**example)
                        yield instance

        def apply_token_indexers(self, instance: Instance) -> None:
            for sentence, mention in zip(
                instance["sentences"].field_list,
                instance["mentions"].field_list,
            ):
                sentence.token_indexers = self._token_indexers
                mention.token_indexers = self._token_indexers
