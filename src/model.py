from tokenizers import Tokenizer
from dotenv import load_dotenv

load_dotenv()

tokenizer = Tokenizer.from_pretrained('gpt2')
