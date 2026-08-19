# ----------------- Imports ------------------ #
from transformer import Transformer, torch, nn
from scheduler import WarmupStableDecayLRScheduler
from tokenizers import Tokenizer

from dotenv import load_dotenv
# -------------------------------------------- #

# -- Environment -- #
load_dotenv()
# ----------------- #

# ------- Configuration ------- #
LEARNING_RATE = 3e-4
MINIMUM_LEARNING_RATE_RATIO = 0.05
WEIGHT_DECAY = 0.1
D_MODEL = 512
NUMBER_OF_TRANSFORMER_BLOCKS = 8
SEQUENCE_LENGTH = 512
NUMBER_OF_HEADS = 8
BATCH_SIZE = 8
MODE_TRANSITION_STEPS = 10000
# ------------------------------#

# --------------- Tokenizer --------------- #
tokenizer = Tokenizer.from_pretrained('gpt2')
vocab = tokenizer.get_vocab_size()
# ----------------------------------------- #

# --------------------- Model Initialization --------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Transformer(
    d_model=D_MODEL,
    vocab_size=vocab,
    num_transformer_blocks=NUMBER_OF_TRANSFORMER_BLOCKS,
    sequence_length=SEQUENCE_LENGTH,
    num_heads=NUMBER_OF_HEADS).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = WarmupStableDecayLRScheduler(
    optimizer,
    mode='W',
    transition_steps=MODE_TRANSITION_STEPS,
    learning_rate=LEARNING_RATE,
    minimum_ratio=MINIMUM_LEARNING_RATE_RATIO,
)

criterion = nn.CrossEntropyLoss()
# ---------------------------------------------------------------- #

# --- Metrics --- #
global_step = 0
tokens_seen = 0
# --------------- #

if __name__ == "__main__":
    ...