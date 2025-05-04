import os
import random
from datasets import load_dataset
import tiktoken
import numpy as np

# 1. Streaming load of English Wikipedia
ds = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True
)

# 2. Prepare tokenizer and output files
enc = tiktoken.get_encoding("gpt2")
script_dir = os.path.dirname(__file__)
train_path = os.path.join(script_dir, "train.bin")
val_path   = os.path.join(script_dir, "val.bin")
train_f = open(train_path, "wb")
val_f   = open(val_path,   "wb")

# 3. Process records one-by-one with random 90/10 split
for idx, example in enumerate(ds):
    text = example["text"]
    ids  = enc.encode_ordinary(text)
    arr  = np.array(ids, dtype=np.uint16)
    # 10% chance to go into validation set
    if random.random() < 0.1:
        val_f.write(arr.tobytes())
    else:
        train_f.write(arr.tobytes())

# 4. Close files
train_f.close()
val_f.close()
print("Finished streaming, tokenizing, and writing train.bin / val.bin.")
