"""Implements CNN encoder as described in the paper "Hierarchical Losses and New Resources for Fine-grained Entity Typing and Linking",
but with extra configurability"""

from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.activations import Activation
import torch
import logging


logger = logging.getLogger(__name__)


@Seq2SeqEncoder.register("cnn1d")
class SingleLayerCNN1d(Seq2SeqEncoder):

    """Single layer 1d CNN as described in the "Hierarchical loss ..." paper.
    Perfroms :class:`torch.nn.Conv1d` on input of shape (batch, input_size, seq_size).

    Note:
        The equation in sec 3.2.2 in the paper does not correspond to applying conv1d because
        In conv1d, sum across w will happen before sum across the channels
        (the matrix multiplication in this equation).

    Todo:
        Write tests
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        context_size: int,
        non_linearity: Activation = None,
        maintain_seq_length: bool = True,
    ) -> None:
        """


        Args:
            input_size: Input size in (batch, input_size, seq_length)
            hidden_size: Output size
            context_size: Context/Kernel size. This should be an odd number.
            non_linearity: Apply a non-linearity to the output of convolution
            maintain_seq_length: Assigns zero padding to always have
                output seq_length same as input.

        Returns: (None)

        Raises:
            ValueError: if context_size is even

        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size

        if (context_size - 1) % 2 != 0:
            raise ValueError(
                f"Context size should be odd but is {context_size}"
            )

        if maintain_seq_length:
            padding: int = (context_size - 1) // 2
        else:
            padding = 0
        self.padding = padding
        self._conv = torch.nn.Conv1d(
            input_size, hidden_size, context_size, padding=padding
        )
        self._non_linearity = non_linearity or Activation.by_name("linear")()

    def is_bidirectional(self) -> bool:
        return False

    def get_output_dim(self) -> int:
        return self.hidden_size

    def get_input_dim(self) -> int:
        return self.input_size

    def get_output_seq_len(self, input_seq_len: int) -> int:
        return input_seq_len + 2 * self.padding - (self.context_size - 1)

    def forward(
        self, token_embeddings: torch.Tensor, mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        """TODO: Docstring for forward.

        Args:
            token_embeddings: TODO
            mask: TODO

        Returns:
            torch.Tensor: Tensor of shape (batch, output_size, output_seq_len)

        """
        # transpose to (batch, input_size, seq_len)
        transposed_token_embeddings = token_embeddings.transpose(1, 2)

        # apply mask

        if mask is not None:
            fill_mask: torch.Tensor = ~mask.unsqueeze(
                1
            )  # shape (batch, 1, seq_len), 1 will be broadcasted by fill
            masked_transposed_token_embeddings = (
                transposed_token_embeddings.masked_fill_(fill_mask, 0.0)
            )
        else:
            masked_transposed_token_embeddings = transposed_token_embeddings

        conv_output = self._non_linearity(
            self._conv(masked_transposed_token_embeddings)
        )  # shape (batch, output_size, out_seq_len),
        # where out_seq_len depends on self.maintain_input_seq_len
        output = conv_output.transpose(1, 2)

        return output
