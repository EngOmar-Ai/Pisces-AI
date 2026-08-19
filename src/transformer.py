from torch import nn
import torch

class MultiLayerPerceptron(nn.Module):
    """
    Feed-forward (MLP) block used inside each transformer block.

    Applies the standard transformer FFN pattern: an up-projection to a wider
    hidden dimension, a GELU nonlinearity, dropout, a down-projection back to
    `d_model`, and a final dropout.
    """

    def __init__(self, d_model: int, d_hidden: int) -> None:
        """
        Initialize the MLP's linear layers.

        Args:
            d_model: Input and output feature dimension (the model's hidden size).
            d_hidden: Dimension of the intermediate hidden layer, typically
                4 * d_model.
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
        Apply the MLP to the input.

        Args:
            x: Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Output tensor of shape (batch, sequence_length, d_model).
        """

        return self.network(x)

class TransformerBlock(nn.Module):
    """
    A single pre-norm transformer decoder block.

    Consists of a self-attention sub-layer followed by an MLP sub-layer, each
    wrapped in a pre-layer-norm + residual connection (i.e. LayerNorm is applied
    before the sub-layer, and its input is added back afterward).
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        """
        Initialize the block's sub-layers.

        Args:
            d_model: Model hidden dimension, must be divisible by `num_heads`.
            num_heads: Number of attention heads used in the multi-head
                self-attention sub-layer.
        """

        super().__init__()

        self.first_layer_norm = nn.LayerNorm(d_model)
        self.multi_head_attention = nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)

        self.second_layer_norm = nn.LayerNorm(d_model)
        self.multi_layer_perceptron = MultiLayerPerceptron(d_model, 4 * d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor|None =None) -> torch.Tensor:
        """
        Run the block's self-attention and MLP sub-layers, each with a residual connection.

        Args:
           x: Input tensor of shape (batch, sequence_length, d_model).
           mask: Optional attention mask passed to `nn.MultiheadAttention` as
               `attn_mask` (e.g. a causal mask of shape
               (sequence_length, sequence_length) with -inf in disallowed
               positions). Defaults to None (no masking).

        Returns:
           Output tensor of shape (batch, sequence_length, d_model).
        """

        output = self.first_layer_norm(x)
        output = self.multi_head_attention(query=output, key=output, value=output, attn_mask=mask)[0]

        residual = x + output

        output = self.second_layer_norm(residual)
        output = self.multi_layer_perceptron(output)

        output = output + residual

        return output

class Transformer(nn.Module):
    """
    A GPT-style decoder-only transformer for autoregressive language modeling.

    Combines learned token and positional embeddings, a stack of causally-masked
    `TransformerBlock`s, a final layer norm, and a linear output head whose
    weights are tied to the token embedding matrix.
    """

    def __init__(self, d_model: int, vocab_size: int, num_heads: int, sequence_length: int, num_transformer_blocks: int) -> None:
        """
        Build the embeddings, causal mask buffer, transformer blocks, and output head.

        Args:
            d_model: Model hidden dimension used throughout the network.
            vocab_size: Number of tokens in the vocabulary, used for both the token embedding table and the output projection.
            num_heads: Number of attention heads in each transformer block.
            sequence_length: Maximum sequence length supported; used to size the positional embedding table and the causal attention mask.
            num_transformer_blocks: Number of stacked `TransformerBlock` layers.

        Notes:
            Registers a lower-triangular causal mask (`-inf` above the diagonal)
            as a buffer named "mask" so it moves with the module across devices.
            Initializes all weights via `initialize_weights`, then ties the
            output layer's weight matrix to the token embedding weight matrix.
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
        Run a forward pass through the transformer to produce next-token logits.

        Args:
            x: Input tensor of token ids with shape (batch, sequence_length),
                where sequence_length must not exceed the `sequence_length`
                the model was initialized with.

        Returns:
            Logits tensor of shape (batch, sequence_length, vocab_size).
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
        Initialize weights for a single module, applied recursively via `self.apply`.

        Args:
            module: A submodule of the `Transformer` (as passed by `nn.Module.apply`).

        Behavior:
            - `nn.Linear` / `nn.Embedding`: weights are initialized from a
              normal distribution with mean 0.0 and std 0.02; `nn.Linear`
              biases (if present) are zeroed.
            - `nn.LayerNorm`: bias is zeroed and weight is filled with 1.0.
            - Any other module type is left unchanged.
        """

        with torch.no_grad():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.normal_(mean=0.0, std=0.02)

                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.zero_()

            elif isinstance(module, nn.LayerNorm):
                module.bias.zero_()
                module.weight.fill_(1.0)

if __name__ == "__main__":
    ...