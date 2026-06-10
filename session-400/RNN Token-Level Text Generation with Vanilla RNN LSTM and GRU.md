# Token-Level Text Generation with PyTorch (Vanilla RNN, LSTM, GRU)

---

# Step 1: Import Libraries

```
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from collections import Counter
from torch.utils.data import Dataset, DataLoader
```

```
print("Libraries loaded.")
print("PyTorch version:", torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
```

---

# Step 2: Load Dataset

```
df = pd.read_csv("./datasets/imdb_balanced_10k.csv")
```

```
print("Dataset size:", len(df))

print("Columns:", df.columns.tolist())

print("\nFirst review preview:")
print(df["text"].iloc[0][:500])
```

---

# Step 3: Reduce Dataset Size

We use part of the dataset for teaching speed.

```
MAX_REVIEWS = 3000

texts = df["text"].astype(str).iloc[:MAX_REVIEWS].tolist()
```

```
print("Reviews used:", len(texts))

print("\nSecond review preview:")
print(texts[1][:300])
```

---

# Step 4: Clean and Tokenize

```
def tokenize(text):
    text = text.lower()

    text = re.sub(r"[^a-z0-9\s']", " ", text)

    tokens = text.split()

    return tokens
```

```
sample_tokens = tokenize(texts[0])

print("First 30 tokens:")
print(sample_tokens[:30])

print("\nToken count:")
print(len(sample_tokens))
```

---

# Step 5: Build Vocabulary

```
all_tokens = []

for text in texts:
    all_tokens.extend(tokenize(text))

word_counts = Counter(all_tokens)
```

```
VOCAB_SIZE = 8000

most_common = word_counts.most_common(VOCAB_SIZE - 2)
```

```
word_to_idx = {
    "<PAD>": 0,
    "<UNK>": 1
}

for idx, (word, _) in enumerate(most_common, start=2):
    word_to_idx[word] = idx

idx_to_word = {
    idx: word
    for word, idx in word_to_idx.items()
}
```

```
print("Vocabulary size:", len(word_to_idx))

print("\nFirst 30 vocabulary words:")
print(list(word_to_idx.keys())[:30])
```

---

# Step 6: Encode Reviews

```
def encode_tokens(tokens):
    return [
        word_to_idx.get(token, word_to_idx["<UNK>"])
        for token in tokens
    ]
```

```
encoded_reviews = []

for text in texts:
    tokens = tokenize(text)

    encoded_reviews.append(
        encode_tokens(tokens)
    )
```

```
print("First encoded review:")
print(encoded_reviews[0][:40])
```

---

# Step 7: Build Sequences

We use 20 tokens to predict the next token.

```
SEQ_LENGTH = 20

X = []
y = []

for review in encoded_reviews:
    if len(review) <= SEQ_LENGTH:
        continue

    for i in range(len(review) - SEQ_LENGTH):
        X.append(review[i:i+SEQ_LENGTH])

        y.append(review[i+SEQ_LENGTH])

X = np.array(X, dtype=np.int64)

y = np.array(y, dtype=np.int64)
```

```
print("X shape:", X.shape)

print("y shape:", y.shape)

print("\nFirst input sequence:")
print(X[0])

print("\nDecoded:")
print([idx_to_word[idx] for idx in X[0]])

print("\nTarget:")
print(idx_to_word[y[0]])
```

---

# Step 8: Reduce Sample Count

```
MAX_SAMPLES = 60000

X = X[:MAX_SAMPLES]

y = y[:MAX_SAMPLES]
```

```
print("Samples used:", len(X))
```

---

# Step 9: Dataset Class

```
class TokenDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)

        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

```
dataset = TokenDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)

print("Batches per epoch:", len(loader))
```

---

# Step 10: Build Generic Token RNN

```
class TokenRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        cell_type="rnn"
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        if cell_type == "rnn":
            self.rnn = nn.RNN(
                embed_dim,
                hidden_dim,
                batch_first=True
            )

        elif cell_type == "lstm":
            self.rnn = nn.LSTM(
                embed_dim,
                hidden_dim,
                batch_first=True
            )

        elif cell_type == "gru":
            self.rnn = nn.GRU(
                embed_dim,
                hidden_dim,
                batch_first=True
            )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x):
        x = self.embedding(x)

        output, hidden = self.rnn(x)

        output = output[:, -1, :]

        logits = self.fc(output)

        return logits
```

```
print("Model class ready.")
```

---

# Step 11: Training Function

```
def train_model(model, loader, epochs=20):
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(epochs):
        total_loss = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)

            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_X)

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}"
        )

    return model
```

```
print("Training function ready.")
```

---

# Step 12: Train Vanilla RNN

```
vanilla_model = TokenRNN(
    vocab_size=len(word_to_idx),
    embed_dim=128,
    hidden_dim=256,
    cell_type="rnn"
)
```

```
vanilla_model = train_model(
    vanilla_model,
    loader
)
```

---

# Step 13: Train LSTM

```
lstm_model = TokenRNN(
    vocab_size=len(word_to_idx),
    embed_dim=128,
    hidden_dim=256,
    cell_type="lstm"
)
```

```
lstm_model = train_model(
    lstm_model,
    loader
)
```

---

# Step 14: Train GRU

```
gru_model = TokenRNN(
    vocab_size=len(word_to_idx),
    embed_dim=128,
    hidden_dim=256,
    cell_type="gru"
)
```

```
gru_model = train_model(
    gru_model,
    loader
)
```

---

# Step 15: Token-Level Text Generation

```
def generate_text(
    model,
    seed_text,
    length=50
):
    model.eval()

    tokens = tokenize(seed_text)

    generated = tokens.copy()

    for _ in range(length):
        encoded = [
            word_to_idx.get(
                token,
                word_to_idx["<UNK>"]
            )
            for token in generated[-SEQ_LENGTH:]
        ]

        if len(encoded) < SEQ_LENGTH:
            encoded = (
                [word_to_idx["<PAD>"]] *
                (SEQ_LENGTH - len(encoded))
            ) + encoded

        x = torch.tensor(
            [encoded],
            dtype=torch.long
        ).to(device)

        with torch.no_grad():
            logits = model(x)

        next_idx = torch.argmax(
            logits,
            dim=1
        ).item()

        next_word = idx_to_word.get(
            next_idx,
            "<UNK>"
        )

        generated.append(next_word)

    return " ".join(generated)
```

```
print("Generation function ready.")
```

---

# Step 16: Generate with Vanilla RNN

```
seed = "this movie was"
```

```
vanilla_output = generate_text(
    vanilla_model,
    seed,
    length=80
)
```

```
print("Vanilla Output:\n")
print(vanilla_output)
```

---

# Step 17: Generate with LSTM

```
lstm_output = generate_text(
    lstm_model,
    seed,
    length=80
)
```

```
print("LSTM Output:\n")
print(lstm_output)
```

---

# Step 18: Generate with GRU

```
gru_output = generate_text(
    gru_model,
    seed,
    length=80
)
```

```
print("GRU Output:\n")
print(gru_output)
```

---

# Step 19: Compare Outputs

```
print("Seed:", seed)

print("\nVanilla:")
print(vanilla_output)

print("\nLSTM:")
print(lstm_output)

print("\nGRU:")
print(gru_output)
```

---

# Step 20: Try Multiple Seeds

```
seeds = [
    "the film was",
    "i loved this",
    "one of the",
    "this movie is"
]
```

```
for seed_text in seeds:
    print(f"\nSeed: {seed_text}")
    print("-" * 60)

    sample = generate_text(
        gru_model,
        seed_text,
        length=40
    )

    print(sample)
```
