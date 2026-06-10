# Task 21: RNN / LSTM / GRU for IMDB Sequence Classification

**Goal**: Build sequence classification models step by step.

---

# Step 1: Import libraries

```python
import re
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
```

---

# Step 2: Setup device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
```

---

# Step 3: Load dataset

```python
df = pd.read_csv(Path.cwd() / "datasets" / "imdb_balanced_10k.csv")

print("Dataset shape:", df.shape)

df.head()
```

---

# Step 4: Clean and tokenize text

```python
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

tokenized_text = [tokenize(text) for text in df["text"]]

print("First sample tokens:")
print(tokenized_text[0][:20])
```

---

# Step 5: Build vocabulary

```python
all_words = [word for sentence in tokenized_text for word in sentence]

vocab = {
    word: i + 2
    for i, (word, _) in enumerate(Counter(all_words).most_common(5000))
}

vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

print("Vocabulary size:", len(vocab))
print("First 10 vocab items:")
print(list(vocab.items())[:10])
```

---

# Step 6: Convert text to sequences

```python
def text_to_seq(text_list, max_len=200):

    seqs = []

    for sentence in text_list:

        seq = [vocab.get(word, 1) for word in sentence]

        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]

        seqs.append(seq)

    return seqs

X = torch.tensor(text_to_seq(tokenized_text), dtype=torch.long)

y = torch.tensor(df["label"].values, dtype=torch.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)
```

---

# Step 7: Split dataset

```python
train_idx = int(0.8 * len(X))

X_train = X[:train_idx]
X_test = X[train_idx:]

y_train = y[:train_idx]
y_test = y[train_idx:]

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=32
)

print("Train samples:", len(X_train))
print("Test samples:", len(X_test))
```

---

# Step 8: Build model

```python
class SequenceClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, model_type="RNN"):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        if model_type == "RNN":
            self.rnn = nn.RNN(
                embed_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=True
            )

        elif model_type == "LSTM":
            self.rnn = nn.LSTM(
                embed_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=True
            )

        elif model_type == "GRU":
            self.rnn = nn.GRU(
                embed_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=True
            )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):

        embedded = self.embedding(x)

        out, _ = self.rnn(embedded)

        pooled = out.mean(dim=1)

        pooled = self.dropout(pooled)

        logits = self.fc(pooled)

        return logits
```

---

# Step 9: Build training function

```python
def train_and_eval(model_type="RNN", epochs=50):

    model = SequenceClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=128,
        model_type=model_type
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.BCEWithLogitsLoss()

    history = []

    print(f"Training {model_type}...")

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x).squeeze()

            loss = criterion(logits, batch_y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5
            )

            optimizer.step()

            total_loss += loss.item()

        model.eval()

        correct = 0

        with torch.no_grad():

            for bx, by in test_loader:

                bx = bx.to(device)
                by = by.to(device)

                probs = torch.sigmoid(
                    model(bx).squeeze()
                )

                preds = (probs > 0.5).float()

                correct += (preds == by).sum().item()

        acc = correct / len(test_loader.dataset)

        history.append(acc)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"Test Acc: {acc:.4f}"
        )

    return history
```

---

# Step 10: Train RNN

```python
rnn_hist = train_and_eval("RNN", epochs=50)
```

---

# Step 11: Train LSTM

```python
lstm_hist = train_and_eval("LSTM", epochs=50)
```

---

# Step 12: Train GRU

```python
gru_hist = train_and_eval("GRU", epochs=50)
```

---

# Step 13: Compare results

```python
plt.figure(figsize=(10, 6))

plt.plot(rnn_hist, label="RNN")
plt.plot(lstm_hist, label="LSTM")
plt.plot(gru_hist, label="GRU")

plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")

plt.title("RNN vs LSTM vs GRU")

plt.legend()

plt.show()
```

---

# Step 14: Print best accuracy

```python
print("Best RNN Accuracy:", max(rnn_hist))
print("Best LSTM Accuracy:", max(lstm_hist))
print("Best GRU Accuracy:", max(gru_hist))
```

