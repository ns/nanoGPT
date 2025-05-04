import os
import json
import requests
import tiktoken
import numpy as np

# 1. Download the Alpaca dataset
DATA_URL = 'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json'
SCRIPT_DIR = os.path.dirname(__file__)
INPUT_JSON = os.path.join(SCRIPT_DIR, 'alpaca_data.json')

if not os.path.exists(INPUT_JSON):
    print(f"Downloading Alpaca dataset to {INPUT_JSON}...")
    resp = requests.get(DATA_URL)
    resp.raise_for_status()
    with open(INPUT_JSON, 'w', encoding='utf-8') as f:
        f.write(resp.text)
    print("Download complete.")

# 2. Load and format examples
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    examples = json.load(f)

formatted_texts = []
for ex in examples:
    instr = ex.get('instruction', '').strip()
    inp   = ex.get('input', '').strip()
    out   = ex.get('output', '').strip()
    # Build the prompt–response string
    text = f"### Instruction:\n{instr}\n"
    if inp:
        text += f"### Input:\n{inp}\n"
    text += f"### Response:\n{out}\n"
    formatted_texts.append(text)

# 3. Split into train/validation by example count
n = len(formatted_texts)
split = int(n * 0.9)
train_texts = formatted_texts[:split]
val_texts   = formatted_texts[split:]

# Concatenate into single strings
train_data = "\n".join(train_texts)
val_data   = "\n".join(val_texts)

# 4. Tokenize with tiktoken GPT-2 BPE
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids   = enc.encode_ordinary(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val   has {len(val_ids):,} tokens")

# 5. Save to binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids   = np.array(val_ids,   dtype=np.uint16)
train_ids.tofile(os.path.join(SCRIPT_DIR, 'train.bin'))
val_ids.tofile(os.path.join(SCRIPT_DIR, 'val.bin'))

print("Wrote train.bin and val.bin.")
