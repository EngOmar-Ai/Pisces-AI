from transformer import Transformer, torch, nn
from tokenizer import vocab

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