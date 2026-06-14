import torch
from bigram import GPTLanguageModel, encode, decode

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')


model = GPTLanguageModel()
model.load_state_dict(torch.load('gpt_model.pth'))
model = model.to(device)
model.eval()
print("Model loaded!")

def generate(prompt, max_new_tokens=200):
    # encode your text into tokens
    encoded = encode(prompt)
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0) # (1, T)

    # generate
    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_new_tokens)

    return decode(output[0].tolist())


while True:
    prompt = input("\n Enter your prompt (or 'quit' to exit):")

    if prompt.lower() == "quit":
        break

    tokens = int(input("How many tokens to generate? (default=200):") or 200)

    result = generate(prompt=prompt, max_new_tokens=tokens)
    print("\n-----Output-----")
    print(result)
    print("--------End-------")