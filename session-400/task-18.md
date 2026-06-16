# Task 18: Random Forest Ensemble

**Goal**: Understand Bagging (Bootstrap Aggregating) in ensemble learning.
**Theory**: Bootstrap Sampling, Voting Mechanism, Feature Importance, Overfitting Mitigation.


```python
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Standalone setup
DATA_DIR = Path.cwd() / "datasets"

# Load the balanced IMDB dataset
df = pd.read_csv(DATA_DIR / "imdb_balanced_10k.csv")

# Vectorize (Using TF-IDF for better performance across multiple trees)
vectorizer = TfidfVectorizer(max_features=2500, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data ready. Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features.")
```


### Step 1: The Overfitting Problem of a Single Tree
A single Decision Tree without depth limits will keep growing until it perfectly memorizes the training data. This leads to **high variance** and poor performance on unseen data (overfitting). Let's prove it.


```python
# 1. Train a fully grown single Decision Tree
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)

# Evaluate on Train and Test
dt_train_acc = accuracy_score(y_train, single_tree.predict(X_train))
dt_test_acc = accuracy_score(y_test, single_tree.predict(X_test))

print(f"Single Tree - Training Accuracy: {dt_train_acc:.4f} (Memorized!)")
print(f"Single Tree - Testing Accuracy:  {dt_test_acc:.4f} (Poor Generalization)")
print(f"Overfitting Gap: {(dt_train_acc - dt_test_acc)*100:.2f}%")
```


### Step 2: Random Forest (Bagging)
Instead of one deep tree, a Random Forest builds $N$ trees (e.g., 100). To ensure the trees are diverse and don't make the exact same mistakes, it uses two tricks:
1. **Bootstrap Sampling**: Each tree is trained on a random sample of the dataset (with replacement).
2. **Feature Randomness**: At each split, the tree can only choose from a random subset of features (usually $\sqrt{M}$).

The final prediction is made by **Majority Voting**.


```python
# 2. Train a Random Forest with 100 trees
# n_jobs=-1 uses all CPU cores for parallel training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluate on Train and Test
rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf_model.predict(X_test))

print(f"Random Forest - Training Accuracy: {rf_train_acc:.4f}")
print(f"Random Forest - Testing Accuracy:  {rf_test_acc:.4f}")
print(f"Overfitting Gap: {(rf_train_acc - rf_test_acc)*100:.2f}%")
print("-" * 40)
print(f"Improvement on Test Data: {(rf_test_acc - dt_test_acc)*100:.2f}%")
```


### Step 3: Feature Importance in the Forest
Even though we can't easily draw 100 trees, we can average the feature importance across all trees to see which words consistently drove the forest's decisions.


```python
# Extract and average feature importances across all 100 trees
importances = rf_model.feature_importances_
feature_names = vectorizer.get_feature_names_out()

# Sort and get top 15
indices = np.argsort(importances)[-15:]

# Visualizing feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], color='tab:blue', align='center', edgecolor='black')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Mean Decrease in Impurity (Feature Importance)')
plt.title('Top 15 Most Important Words in Random Forest (100 Trees)')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
```

