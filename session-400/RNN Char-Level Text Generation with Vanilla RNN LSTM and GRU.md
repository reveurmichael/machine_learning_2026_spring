# RNN Text Generation with PyTorch (Vanilla RNN, LSTM, GRU)

---

# Step 1: Import Libraries

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
```

```python
print("Libraries loaded.")
print("PyTorch version:", torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
```

---

# Step 2: Load Dataset

```python
df = pd.read_csv("./datasets/imdb_balanced_10k.csv")
```

```python
print("Dataset size:", len(df))

print("Columns:", df.columns.tolist())

print("\nFirst review preview:")
print(df["text"].iloc[0][:500])

print("\nFirst label:")
print(df["label"].iloc[0])
```

---

# Step 3: Build Corpus

We keep part of the dataset for faster training.

```python
MAX_REVIEWS = 2000

texts = df["text"].astype(str).iloc[:MAX_REVIEWS].tolist()

corpus = " ".join(texts).lower()
```

```python
print("Reviews used:", len(texts))

print("Corpus length:", len(corpus))

print("\nFirst 500 characters:")
print(corpus[:500])
```

---

# Step 4: Build Character Vocabulary

```python
chars = sorted(list(set(corpus)))

char_to_idx = {
    ch: idx
    for idx, ch in enumerate(chars)
}

idx_to_char = {
    idx: ch
    for ch, idx in char_to_idx.items()
}
```

```python
print("Unique characters:", len(chars))

print("\nFirst 30 characters:")
print(chars[:30])
```

---

# Step 5: Encode Corpus

```python
encoded = np.array([
    char_to_idx[ch]
    for ch in corpus
], dtype=np.int64)
```

```python
print("Encoded length:", len(encoded))

print("First 100 values:")
print(encoded[:100])
```

---

# Step 6: Build Sequences

```python
SEQ_LENGTH = 80

X = []
y = []

for i in range(len(encoded) - SEQ_LENGTH):
    X.append(encoded[i:i+SEQ_LENGTH])
    y.append(encoded[i+SEQ_LENGTH])

X = np.array(X, dtype=np.int64)

y = np.array(y, dtype=np.int64)
```

```python
print("X shape:", X.shape)

print("y shape:", y.shape)

print("\nFirst target character:")
print(idx_to_char[y[0]])
```

---

# Step 7: Reduce Sample Count

```python
MAX_SAMPLES = 40000

X = X[:MAX_SAMPLES]

y = y[:MAX_SAMPLES]
```

```python
print("Samples used:", len(X))
```

---

# Step 8: Build Dataset Class

```python
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

```python
dataset = TextDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)

print("Batches per epoch:", len(loader))
```

---

# Step 9: Build Generic RNN Model

```python
class CharRNN(nn.Module):
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

        self.cell_type = cell_type

    def forward(self, x):
        x = self.embedding(x)

        output, hidden = self.rnn(x)

        output = output[:, -1, :]

        logits = self.fc(output)

        return logits
```

```python
print("Model class ready.")
```

---

# Step 10: Build Training Function

```python
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

```python
print("Training function ready.")
```

---

# Step 11: Train Vanilla RNN

```python
vanilla_model = CharRNN(
    vocab_size=len(chars),
    embed_dim=64,
    hidden_dim=128,
    cell_type="rnn"
)
```

```python
vanilla_model = train_model(
    vanilla_model,
    loader
    )
```

---

# Step 12: Train LSTM

```python
lstm_model = CharRNN(
    vocab_size=len(chars),
    embed_dim=64,
    hidden_dim=128,
    cell_type="lstm"
)
```

```python
lstm_model = train_model(
    lstm_model,
    loader
)
```

---

# Step 13: Train GRU

```python
gru_model = CharRNN(
    vocab_size=len(chars),
    embed_dim=64,
    hidden_dim=128,
    cell_type="gru"
)
```

```python
gru_model = train_model(
    gru_model,
    loader
)
```

---

# Step 14: Build Text Generator

```python
def generate_text(
    model,
    seed_text,
    length=400
):
    model.eval()

    generated = seed_text.lower()

    for _ in range(length):
        sequence = [
            char_to_idx.get(ch, 0)
            for ch in generated[-SEQ_LENGTH:]
        ]

        if len(sequence) < SEQ_LENGTH:
            sequence = [0] * (SEQ_LENGTH - len(sequence)) + sequence

        x = torch.tensor(
            [sequence],
            dtype=torch.long
        ).to(device)

        with torch.no_grad():
            logits = model(x)

        next_idx = torch.argmax(
            logits,
            dim=1
        ).item()

        next_char = idx_to_char[next_idx]

        generated += next_char

    return generated
```

```python
print("Generation function ready.")
```

---

# Step 15: Generate Text with Vanilla RNN

```python
seed = "this movie"
```

```python
vanilla_output = generate_text(
    vanilla_model,
    seed,
    length=500
)
```

```python
print("Vanilla RNN Output:\n")
print(vanilla_output)
```

---

# Step 16: Generate Text with LSTM

```python
lstm_output = generate_text(
    lstm_model,
    seed,
    length=500
)
```

```python
print("LSTM Output:\n")
print(lstm_output)
```

---

# Step 17: Generate Text with GRU

```python
gru_output = generate_text(
    gru_model,
    seed,
    length=500
)
```

```python
print("GRU Output:\n")
print(gru_output)
```

---

# Step 18: Compare Outputs

```python
print("Seed:", seed)

print("\nVanilla Sample:")
print(vanilla_output[:300])

print("\nLSTM Sample:")
print(lstm_output[:300])

print("\nGRU Sample:")
print(gru_output[:300])
```

---

# Step 19: Try Different Seeds

```python
seeds = [
    "the movie",
    "i loved",
    "this film",
    "one of the"
]
```

```python
for seed_text in seeds:
    print(f"\nSeed: {seed_text}")
    print("-" * 60)

    sample = generate_text(
        gru_model,
        seed_text,
        length=200
    )

    print(sample)
```
