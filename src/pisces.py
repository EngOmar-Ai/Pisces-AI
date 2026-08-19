from model import model, tokenizer

# ------------------------------------ Invoke Algorithm ------------------------------------ #
# 1) Tokenize The Prompt To Get The Sequence Of Tokens (SEQ)
# 2) Create A Tensor And Unsqueeze To Get The Required Shape (1, SEQ)
# 3) Pass It To The Model And Get Its Predictions, Should Be Of Shape (1, SEQ, VOCAB)
# 4) Get The Logits Of The Very Last Token, Should Be In The Shape (VOCAB)
# 5) Get The Token With The Highest Probability
# 6) Append This Token To The Original Prompt And ReInvoke The Model With The Updated Prompt
# 7) Keep Appending Tokens Until Reach A Set Number Of Tokens Or Reach <|endoftext|> Token
# ------------------------------------------------------------------------------------------ #

def invoke(prompt: str) -> str:
    ...

if __name__ == '__main__':
    print("| ------- Initializing Conversation With Pisces Ai ------- |")
    user = input("User: ")
    print()
    assistant = invoke("User: " + user)
