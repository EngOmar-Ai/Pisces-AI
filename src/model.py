# ====================== #
#         Imports        #
# ====================== #

from tokenizers import Tokenizer
from dotenv import load_dotenv

import os

from torch import nn
import torch

# =============================== #
#        Load '.env' File         #
# =============================== #

load_dotenv()

# ===================================================== #
#        Define The Multi-Layer Perceptron Class        #
# ===================================================== #

class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_model: int, d_hidden: int) -> None:
        """
        Feed-forward network used inside a Transformer block.

        Expands the model dimension by a factor of four, applies GELU
        activation and dropout, then projects the representation back
        to the original model dimension.

        Args:
            d_model: Dimensionality of the Transformer representations.
            d_hidden: Hidden dimensionality of the feed-forward network.

        Input shape:
            [batch_size, sequence_length, d_model]

        Output shape:
            [batch_size, sequence_length, d_model]
        """

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the input through the Multi Layer Perceptron.

        Args:
            x: Input sequence representations.

        Returns:
            The Processed sequence representations.
        """

        return self.network(x)

# ================================================ #
#        Define The Transformer Block Class        #
# ================================================ #

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        """
        A single pre-layer-normalized Transformer block.

        The block consists of a multi-head self-attention layer followed
        by a feed-forward network. Layer normalization is applied before
        each sublayer, and residual connections are applied after each
        sublayer.

        Args:
            d_model: Dimensionality of the Transformer representations.
            num_heads: Number of attention heads.

        Input shape:
            [batch_size, sequence_length, d_model]

        Output shape:
            [batch_size, sequence_length, d_model]
        """

        super().__init__()

        self.first_layer_norm = nn.LayerNorm(d_model)
        self.multi_head_attention = nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)

        self.second_layer_norm = nn.LayerNorm(d_model)
        self.multi_layer_perceptron = MultiLayerPerceptron(d_model, 4 * d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor|None =None) -> torch.Tensor:
        """
        Run the input through the Transformer block.

        Args:
            x: Input sequence representations.
            mask: Optional causal attention mask. Positions masked with
                negative infinity cannot be attended to.

        Returns:
            The transformed sequence representations.
        """

        output = self.first_layer_norm(x)
        output = self.multi_head_attention(query=output, key=output, value=output, attn_mask=mask)[0]

        residual = x + output

        output = self.second_layer_norm(residual)
        output = self.multi_layer_perceptron(output)

        output = output + residual

        return output

# ========================================== #
#        Define The Transformer Class        #
# ========================================== #

class Transformer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, num_heads: int, sequence_length: int, num_transformer_blocks: int) -> None:
        """
        GPT-style Transformer language model.

        Converts token IDs into token and positional embeddings, processes
        them through a stack of causal Transformer blocks, and projects the
        final representations into vocabulary logits.

        The token embedding weights are shared with the output projection
        layer (weight tying), reducing the number of trainable parameters
        and allowing the same token representations to be used for both
        input embeddings and output predictions.

        Args:
            d_model: Dimensionality of the Transformer representations.
            vocab_size: Number of tokens in the vocabulary.
            num_heads: Number of attention heads in each Transformer block.
            sequence_length: Maximum supported sequence length.
            num_transformer_blocks: Number of Transformer blocks.

        Input shape:
            [batch_size, sequence_length]

        Output shape:
            [batch_size, sequence_length, vocab_size]

        Note:
            The model uses a causal attention mask so that each token can
            only attend to itself and preceding tokens.
        """

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
        """
        Perform a forward pass through the Transformer language model.

        Token IDs are converted into token and positional embeddings, then
        passed through the Transformer blocks using a causal attention mask.
        The final representations are normalized and projected into logits
        over the vocabulary.

        Args:
            x: Input token IDs with shape
                [batch_size, sequence_length].

        Returns:
            Logits for each token position with shape
            [batch_size, sequence_length, vocab_size].
        """

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
        """
        Initialize trainable parameters using GPT-style initialization.

        Linear and embedding weights are initialized from a normal
        distribution with mean 0 and standard deviation 0.02. Linear
        biases are initialized to zero, while LayerNorm weights and
        biases are initialized to one and zero respectively.

        Args:
            module: Module whose parameters should be initialized.
        """

        with torch.no_grad():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.normal_(mean=0.0, std=0.02)

                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.zero_()

            elif isinstance(module, nn.LayerNorm):
                module.bias.zero_()
                module.weight.fill_(1.0)

# ========================================================== #
#        Define The Model And Its Hyper Parameters         #
# ========================================================== #

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


# ============================= #
#             Main              #
# ============================= #

if __name__ == "__main__":
    ...