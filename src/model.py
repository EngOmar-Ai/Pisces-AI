from tokenizers import Tokenizer
from dotenv import load_dotenv

load_dotenv()

from torch import nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_model, d_hidden):
        super(MultiLayerPerceptron, self).__init__()

        self.Network = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.Network(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super(TransformerBlock, self).__init__()

        self.FirstLayerNorm = nn.LayerNorm(d_model)
        self.MultiHeadAttention = nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)
        self.SecondLayerNorm = nn.LayerNorm(d_model)
        self.MultiLayerPerceptron = MultiLayerPerceptron(d_model, 4*d_model)

    def forward(self, x, mask=None):
        output = self.FirstLayerNorm(x)
        output = self.MultiHeadAttention(query=output,key=output,value=output,attn_mask=mask)[0]
        residual = x + output
        output = self.SecondLayerNorm(residual)
        output = self.MultiLayerPerceptron(output)
        output = output + residual

        return output


tokenizer = Tokenizer.from_pretrained('gpt2')
vocab = tokenizer.get_vocab_size()
