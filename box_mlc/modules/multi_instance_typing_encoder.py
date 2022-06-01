"""Common encoding sequence for multi-instance entity typing implemented as a module"""
from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.common import Registrable
from allennlp.modules.seq2seq_encoders import (
    Seq2SeqEncoder,
    PassThroughEncoder,
)
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import torch
import logging

logger = logging.getLogger(__name__)


class MultiInstanceTypingEncoder(torch.nn.Module, Registrable):
    """Base class to satisfy Registrable"""

    default_implementation = "multi-instance-typing-encoder"

    pass


@MultiInstanceTypingEncoder.register("multi-instance-typing-encoder")
class GenericMultiInstanceTypingEncoder(MultiInstanceTypingEncoder):
    def __init__(
        self,
        textfield_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder = None,
        seq2seq_encoder: Seq2SeqEncoder = None,
        debug_level: int = 0,
        **kwargs: Any,
    ) -> None:
        """

        Args:
            textfield_embedder: TODO
            seq2seq_encoder: TODO
            seq2vec_encoder: TODO
            mention_seq2seq_encoder: None for no seq2seq for mention,
                "same" for using same seq2seq_encoder as the sentences,
                Seq2SeqEncoder params for a separate instance of a Seq2SeqEncoder
            mention_seq2vec_encoder: Same as mention_seq2seq_encoder
            debug_level: scale of 0 to 3
            **kwargs: TODO

        Returns: (None)

        """
        super().__init__()  # type:ignore
        self._textfield_embedder = textfield_embedder
        self._seq2vec_encoder = seq2vec_encoder  # type: ignore

        self._seq2seq_encoder: Optional[Seq2SeqEncoder] = (
            seq2seq_encoder
        )

        self.debug_level = debug_level

    @property
    def sentences_output_order(self) -> int:
        if self._seq2vec_encoder is None:
            return 3
        else:
            return 2

    def forward(
        self, text: TextFieldTensors
    ) -> torch.Tensor:
        """Applies the sequence of textfield_embedder->seq2seq_encoder->seq2vec_encoder
        to both sentences and mentions assuming that they are formed from `ListField[TextField]`.

        Args:
            sentences: TODO
            mentions: TODO

        Returns:
            sentences_vec: Vector of shape (batch, num_sentences,hidden_size) or (batch, num_sentences, seq_len, hidden_size)
            mention_vec: Vector of shape (batch, num_sentences,hidden_size) or (batch, num_sentences, mention_seq_len, hidden_size)

        """
        # get the embeddings assuming that shape of tensor is (batch, sentences, seq_len)
        text_emb = self._textfield_embedder(
            text
        )
        # sentences_emb = self._textfield_embedder(
        #     sentences, num_wrapping_dims=1
        # )  # shape (batch, sentences, seq_len, hidden_dim)
        text_mask = get_text_field_mask(
            text
        )  # shape (batch, sentences, seq_len)

        # skip seq2seq_encoder if pass-through

        sentences_s2s = (
            self._seq2seq_encoder(text_emb, text_mask)
            if self._seq2seq_encoder is not None
            else text_emb
        )

        sentences_vec = (
            self._seq2vec_encoder(sentences_s2s, text_mask)
            if self._seq2vec_encoder is not None
            else sentences_s2s
        )

        return sentences_vec
