import torch
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import masked_mean


@Seq2VecEncoder.register("mean")
class MeanSeq2VecEncoder(Seq2VecEncoder):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def get_input_dim(self) -> int:
        return self.embedding_dim

    def get_output_dim(self) -> int:
        return self.embedding_dim

    def __call__(self, vector: torch.Tensor, mask: torch.BoolTensor):
        return masked_mean(vector, mask.unsqueeze(-1), dim=1)
