from tokenizers import Tokenizer
from dotenv import load_dotenv

load_dotenv()

tokenizer = Tokenizer.from_pretrained('gpt2')
vocab = tokenizer.get_vocab_size()

if __name__ == "__main__":
    ...