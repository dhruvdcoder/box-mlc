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
import allennlp
from wcmatch import glob

if sys.version_info >= (3, 8):
    from typing import (
        TypedDict,
    )  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict, Literal, overload

import logging
import json
import numpy as np
from overrides import overrides
from allennlp.common.file_utils import cached_path
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
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from .common import JSONTransform

logger = logging.getLogger(__name__)


class Mention(TypedDict):
    """
    Single entry in a mention bag.
    """

    entity: str  #: entity name. Ex: "Harry_Potter"
    sentence: str  #: A single sentence containing a mention. Ex: "My Immortal is a Harry Potter fan fiction serially published on FanFiction"
    sf_mention: str  #: surface form of the mention. Ex: "Harry Potter"
    title: str  #: not important. Keeping it as it was present in the data. Ex: "title:My Immortal (fan fiction)"


class InstanceFields(TypedDict):
    """Contents which form an instance"""

    sentences: ListField  #: it is actually ListField[TextField], one TextField instance per sentence
    mentions: ListField  #: again ListField[TextField]
    labels: MultiLabelField  #: types


class TokenWithPosition(Token):
    __slots__ = ["position"]

    position: Optional[
        str
    ]  #: string because integers will be treated as indices but we have negative values

    def __init__(self, text: str = None, position: str = None):
        super().__init__(text=text)
        self.position = position

    def __str__(self) -> str:
        return f"{self.text}, {self.position}"

    def __repr__(self) -> str:
        return self.__str__()


def _matches(source: List[Token], target: List[Token]) -> bool:
    for s, t in zip(source, target):
        if s.text != t.text:
            return False

    return True


class DataError(Exception):
    pass


allennlp_major_version = int(allennlp.__version__.split(".")[0])


if allennlp_major_version < 2:

    @DatasetReader.register("multi-instance-entity-typing")
    class MultiInstanceEntityTyping(DatasetReader):
        """
        Reader for reading a multi-instance entity typing dataset like in section 4.2
        of paper "Hierarchical Losses and New Resources for Fine-grained Entity Typing and Linking".

        We add relative positions of tokens w.r.t to the mention if needed. This is needed to implement
        the baseline of "Hierarchical losses ..." paper (section 3.2.1).
        """

        def __init__(
            self,
            tokenizer: Tokenizer,
            token_indexers: Dict[str, TokenIndexer],
            add_position_features: bool = True,
            use_transitive_closure: bool = False,
            bag_size: int = 10,
            **kwargs: Any,
        ) -> None:
            """
            Arguments:
                tokenizer: The tokenizer to be used.
                token_indexers: The token_indexers to be used--one per embedder. Will usually be only one.
                add_position_features: Whether to create relative position feature after tokenizing or not.
                use_transitive_closure: use types_extra
                bag_size: number of mentions to take from the bag of each entity
                **kwargs: Parent class args.
                    See https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_reader.py

            Raises:
                ValueError: If bag size > 20
            """
            super().__init__(**kwargs)
            self._tokenizer = tokenizer
            self._token_indexers = token_indexers
            self._add_position_features = add_position_features
            # make sure that if we add position features
            # we have a token indexer for it

            if add_position_features and (len(token_indexers) <= 1):
                raise ValueError(
                    "If add_position_features is set, there should be at least two token_indexers"
                    "One for token text and one for position"
                )
            self._use_transitive_closure = use_transitive_closure

            if bag_size > 20:  # smallest bag in data has 20 mentions
                raise ValueError(
                    f"Maximum bag size should be <=20 but is {bag_size}"
                )
            self._bag_size = bag_size

        def process_single_mention(
            self, sentence: str, mention: str
        ) -> Tuple[TextField, TextField]:
            """Creates :class:`TextFields` for single mention. Also creates position features
            based on parameters passed to the constructor.

            Args:
                sentence: raw input sentence
                mention: raw mention string

            Returns:
                Field for sentence

                Field for mention string

            """
            sentence_tokens = self._tokenizer.tokenize(sentence)
            mention_tokens = self._tokenizer.tokenize(mention)

            if self._add_position_features:
                positions = self.position_features(
                    sentence_tokens, mention_tokens
                )
                sentence_tokens = [
                    TokenWithPosition(text=t.text, position=str(p))
                    for t, p in zip(sentence_tokens, positions)
                ]
                mention_tokens = [
                    TokenWithPosition(text=t.text, position=str(0))
                    for t in mention_tokens
                ]

            return (
                TextField(
                    sentence_tokens, token_indexers=self._token_indexers
                ),
                TextField(mention_tokens, token_indexers=self._token_indexers),
            )

        def example_to_fields(
            self,
            mention_bag: List[Mention],
            types: List[str],
            types_extra: List[str],
            meta: Dict = None,
            **kwargs: Any,
        ) -> InstanceFields:
            """Converts a dictionary containing an example datapoint to fields that can be used
            to create an :class:`Instance`. If a meta dictionary is passed, then it also adds raw data in
            the meta dict.

            Args:
                mention_bag: list of mentions
                types: labels for the typing task
                types_extra: additional labels when TC is used
                meta: TODO
                **kwargs: TODO

            Returns:
                dictionary of fields

            """

            if meta is None:
                meta = {}
            meta["input"] = mention_bag
            meta["types"] = types
            meta["types_extra"] = types_extra
            meta["using_tc"] = False

            (sentence_fields, mention_fields,) = [
                ListField(list(field_list))  # type:ignore
                for field_list in zip(
                    *[
                        self.process_single_mention(
                            mention["sentence"], mention["sf_mention"]
                        )
                        for mention in mention_bag[: self._bag_size]
                    ]
                )
            ]

            if self._use_transitive_closure:
                types += types_extra
                meta["using_tc"] = True
            labels = MultiLabelField(types)

            return {
                "sentences": sentence_fields,
                "mentions": mention_fields,
                "labels": labels,
            }

        def position_features(
            self, text_tokens: List[Token], mention_tokens: List[Token]
        ) -> List[int]:
            """Creates relative position features for each token as stated in section 3.2.1 of the "Hierarchical losses.."
            paper

            Args:
                text_tokens: TODO
                mention_tokens: TODO

            Returns:
                list of integers of relative positions

            Raises:
                DataError: If mention is not found in sentence

            """
            # locate the positions of the mention
            start = 0
            end = 0
            found = False

            for pos, sentence_token in enumerate(text_tokens):
                if mention_tokens[0].text == sentence_token.text:
                    if _matches(
                        text_tokens[pos : pos + len(mention_tokens)],
                        mention_tokens,
                    ):
                        start = pos
                        end = pos + len(mention_tokens)
                        found = True

                        break

            if not found:
                raise DataError(
                    f"mention {' '.join(t.text for t in mention_tokens)} "  # type:ignore
                    f"not present in sentence {' '.join(t.text for t in text_tokens)}"
                )
            positions = (
                list(range(-start, 0, 1))
                + list([0] * len(mention_tokens))
                + list(range(1, len(text_tokens) - (end - 1), 1))
            )
            assert len(positions) == len(text_tokens)

            return positions

        def text_to_instance(  # type:ignore
            self,
            mention_bag: List[Mention],
            types: List[str],
            types_extra: List[str],
            **kwargs: Any,
        ) -> Instance:
            """Converts contents of a single raw example to :class:`Instance`.

            Args:
                mention_bag: list of mentions
                types: labels for the typing task
                types_extra: additional labels when TC is used
                **kwargs: unused

            Returns:
                 :class:`Instance` of data

            Raises:
                DataError: on any kind of data inconsistency, ,


            """
            meta_dict: Dict = {}
            try:
                main_fields = self.example_to_fields(
                    mention_bag, types, types_extra, meta=meta_dict
                )
            except DataError as de:
                if "entity" in kwargs:
                    raise DataError(
                        f"Error in processing entity {kwargs['entity']}"
                    ) from de
                else:
                    raise DataError(
                        f"Error in example\n{dict(mentaion_bag=mention_bag)}"
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

    @JSONTransform.register("from-typenet")
    class FromTypenet(JSONTransform):
        def __call__(self, inp: Dict) -> Dict:
            labels_: List[str] = list(set(types + types_extra))

            inp["labels"] = labels_

            return inp


elif allennlp_major_version >= 2:

    @DatasetReader.register("multi-instance-entity-typing")
    class MultiInstanceEntityTyping(DatasetReader):
        """
        Reader for reading a multi-instance entity typing dataset like in section 4.2
        of paper "Hierarchical Losses and New Resources for Fine-grained Entity Typing and Linking".

        We add relative positions of tokens w.r.t to the mention if needed. This is needed to implement
        the baseline of "Hierarchical losses ..." paper (section 3.2.1).
        """

        def __init__(
            self,
            tokenizer: Tokenizer,
            token_indexers: Dict[str, TokenIndexer],
            add_position_features: bool = True,
            use_transitive_closure: bool = False,
            bag_size: int = 10,
            **kwargs: Any,
        ) -> None:
            """
            Arguments:
                tokenizer: The tokenizer to be used.
                token_indexers: The token_indexers to be used--one per embedder. Will usually be only one.
                add_position_features: Whether to create relative position feature after tokenizing or not.
                use_transitive_closure: use types_extra
                bag_size: number of mentions to take from the bag of each entity
                **kwargs: Parent class args.
                    See https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_reader.py

            Raises:
                ValueError: If bag size > 20
            """
            super().__init__(
                manual_distributed_sharding=True,
                manual_multiprocess_sharding=True,
                **kwargs,
            )
            self._tokenizer = tokenizer
            self._token_indexers = token_indexers
            self._add_position_features = add_position_features
            # make sure that if we add position features
            # we have a token indexer for it

            if add_position_features and (len(token_indexers) <= 1):
                raise ValueError(
                    "If add_position_features is set, there should be at least two token_indexers"
                    "One for token text and one for position"
                )
            self._use_transitive_closure = use_transitive_closure

            if bag_size > 20:  # smallest bag in data has 20 mentions
                raise ValueError(
                    f"Maximum bag size should be <=20 but is {bag_size}"
                )
            self._bag_size = bag_size

        def process_single_mention(
            self, sentence: str, mention: str
        ) -> Tuple[TextField, TextField]:
            """Creates :class:`TextFields` for single mention. Also creates position features
            based on parameters passed to the constructor.

            Args:
                sentence: raw input sentence
                mention: raw mention string

            Returns:
                Field for sentence

                Field for mention string

            """
            sentence_tokens = self._tokenizer.tokenize(sentence)
            mention_tokens = self._tokenizer.tokenize(mention)

            if self._add_position_features:
                positions = self.position_features(
                    sentence_tokens, mention_tokens
                )
                sentence_tokens = [
                    TokenWithPosition(text=t.text, position=str(p))
                    for t, p in zip(sentence_tokens, positions)
                ]
                mention_tokens = [
                    TokenWithPosition(text=t.text, position=str(0))
                    for t in mention_tokens
                ]

            return (
                TextField(
                    sentence_tokens,  # No token indexer here
                ),
                TextField(
                    mention_tokens,  # No token indexer here
                ),
            )

        def example_to_fields(
            self,
            mention_bag: List[Mention],
            types: List[str],
            types_extra: List[str],
            meta: Dict = None,
            **kwargs: Any,
        ) -> InstanceFields:
            """Converts a dictionary containing an example datapoint to fields that can be used
            to create an :class:`Instance`. If a meta dictionary is passed, then it also adds raw data in
            the meta dict.

            Args:
                mention_bag: list of mentions
                types: labels for the typing task
                types_extra: additional labels when TC is used
                meta: TODO
                **kwargs: TODO

            Returns:
                dictionary of fields

            """

            if meta is None:
                meta = {}
            meta["input"] = mention_bag
            meta["types"] = types
            meta["types_extra"] = types_extra
            meta["using_tc"] = False

            (sentence_fields, mention_fields,) = [
                ListField(list(field_list))  # type:ignore
                for field_list in zip(
                    *[
                        self.process_single_mention(
                            mention["sentence"], mention["sf_mention"]
                        )
                        for mention in mention_bag[: self._bag_size]
                    ]
                )
            ]

            if self._use_transitive_closure:
                types += types_extra
                meta["using_tc"] = True
            labels = MultiLabelField(types)

            return {
                "sentences": sentence_fields,
                "mentions": mention_fields,
                "labels": labels,
            }

        def position_features(
            self, text_tokens: List[Token], mention_tokens: List[Token]
        ) -> List[int]:
            """Creates relative position features for each token as stated in section 3.2.1 of the "Hierarchical losses.."
            paper

            Args:
                text_tokens: TODO
                mention_tokens: TODO

            Returns:
                list of integers of relative positions

            Raises:
                DataError: If mention is not found in sentence

            """
            # locate the positions of the mention
            start = 0
            end = 0
            found = False

            for pos, sentence_token in enumerate(text_tokens):
                if mention_tokens[0].text == sentence_token.text:
                    if _matches(
                        text_tokens[pos : pos + len(mention_tokens)],
                        mention_tokens,
                    ):
                        start = pos
                        end = pos + len(mention_tokens)
                        found = True

                        break

            if not found:
                raise DataError(
                    f"mention {' '.join(t.text for t in mention_tokens)} "  # type:ignore
                    f"not present in sentence {' '.join(t.text for t in text_tokens)}"
                )
            positions = (
                list(range(-start, 0, 1))
                + list([0] * len(mention_tokens))
                + list(range(1, len(text_tokens) - (end - 1), 1))
            )
            assert len(positions) == len(text_tokens)

            return positions

        def text_to_instance(  # type:ignore
            self,
            mention_bag: List[Mention],
            types: List[str],
            types_extra: List[str],
            **kwargs: Any,
        ) -> Instance:
            """Converts contents of a single raw example to :class:`Instance`.

            Args:
                mention_bag: list of mentions
                types: labels for the typing task
                types_extra: additional labels when TC is used
                **kwargs: unused

            Returns:
                 :class:`Instance` of data

            Raises:
                DataError: on any kind of data inconsistency, ,


            """
            meta_dict: Dict = {}
            try:
                main_fields = self.example_to_fields(
                    mention_bag, types, types_extra, meta=meta_dict
                )
            except DataError as de:
                if "entity" in kwargs:
                    raise DataError(
                        f"Error in processing entity {kwargs['entity']}"
                    ) from de
                else:
                    raise DataError(
                        f"Error in example\n{dict(mentaion_bag=mention_bag)}"
                    )

            return Instance(
                {**cast(dict, main_fields), "meta": MetadataField(meta_dict)}
            )

        def apply_token_indexers(self, instance: Instance) -> None:
            for sentence, mention in zip(
                instance["sentences"].field_list,
                instance["mentions"].field_list,
            ):
                sentence.token_indexers = self._token_indexers
                mention.token_indexers = self._token_indexers

        def _read(self, file_path: str) -> Iterator[Instance]:
            """Reads a datafile to produce instances

            Args:
                file_path: Path to jsonl file

            Yields:
                data instances

            """
            # with open(file_path) as f:
            #     for line in self.shard_iterable(f.readlines()):
            #         example = json.loads(line)
            #         instance = self.text_to_instance(**example)

            #         yield instance

            for file_ in glob.glob(file_path, flags=glob.EXTGLOB):
                # logger.info(f"Reading {file_}")
                with open(file_) as f:
                    for line in self.shard_iterable(f):
                        example = json.loads(line)
                        instance = self.text_to_instance(**example)
                        yield instance
