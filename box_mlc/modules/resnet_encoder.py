import torch
from torch import nn
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from torchvision.models import resnet


@Seq2VecEncoder.register("resnet50")
class ResNet50Seq2VecEncoder(Seq2VecEncoder):

    def __init__(self, embedding_dim: int, output_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        model = resnet.resnet50(pretrained=True)
        model.fc = nn.Linear(embedding_dim, output_dim)
        self.model = model

    def get_input_dim(self) -> int:
        return self.embedding_dim

    def get_output_dim(self) -> int:
        return self.embedding_dim

    def __call__(self, input: torch.Tensor):
        return self.model(input)
