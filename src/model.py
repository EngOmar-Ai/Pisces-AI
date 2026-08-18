from tokenizers import Tokenizer
from dotenv import load_dotenv

import os

from torch import nn
import torch

load_dotenv()

class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_model: int, d_hidden: int) -> None:

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:

        super().__init__()

        self.first_layer_norm = nn.LayerNorm(d_model)
        self.multi_head_attention = nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)

        self.second_layer_norm = nn.LayerNorm(d_model)
        self.multi_layer_perceptron = MultiLayerPerceptron(d_model, 4 * d_model)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:

        output = self.first_layer_norm(x)
        output = self.multi_head_attention(query=output, key=output, value=output, attn_mask=mask)[0]

        residual = x + output

        output = self.second_layer_norm(residual)
        output = self.multi_layer_perceptron(output)

        output = output + residual

        return output

class Transformer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, num_heads: int, sequence_length: int, num_transformer_blocks: int) -> None:

        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(sequence_length, d_model)

        self.dropout = nn.Dropout(0.1)

        mask = torch.triu(torch.full((sequence_length, sequence_length), float('-inf')), diagonal=1)
        self.register_buffer("mask", mask)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_transformer_blocks)])

        self.layer_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

        self.apply(self.initialize_weights)

        self.output_layer.weight = self.token_embeddings.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        token_embeddings = self.token_embeddings(x)
        position_embeddings = self.position_embeddings(torch.arange(x.shape[1], device=x.device)).unsqueeze(0)
        transformer_input = token_embeddings + position_embeddings

        transformer_output = self.dropout(transformer_input)

        mask = self.mask[:x.shape[1], :x.shape[1]]
        for transformer_block in self.transformer_blocks:
            transformer_output = transformer_block(transformer_output, mask=mask)

        transformer_output = self.layer_norm(transformer_output)
        transformer_output = self.output_layer(transformer_output)

        return transformer_output

    @staticmethod
    def initialize_weights(module):
        with torch.no_grad():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.normal_(mean=0.0, std=0.02)

                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.zero_()

            elif isinstance(module, nn.LayerNorm):
                module.bias.zero_()
                module.weight.fill_(1.0)

tokenizer = Tokenizer.from_pretrained('gpt2')
vocab = tokenizer.get_vocab_size()

database_configuration = {
    "database": os.getenv("DB_NAME"),
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Transformer(512, vocab, 8, 512, 8).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.1
)

scheduler = ...

criterion = nn.CrossEntropyLoss()

batch_size = 8

path = r'../results/Transformer.pth'

if __name__ == "__main__":
    ...