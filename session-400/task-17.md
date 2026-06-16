# Task 17: Decision Trees for Text

**Goal**: Understand explainable hierarchical classification.

```python
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Standalone setup
DATA_DIR = Path.cwd() / "datasets"

# Load the balanced IMDB dataset
df = pd.read_csv(DATA_DIR / "imdb_balanced_10k.csv")

# We use CountVectorizer (Binary=True) to simplify: Does the word exist or not?
vectorizer = CountVectorizer(max_features=1000, stop_words='english', binary=True)
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset vectorized. Vocabulary size: {len(vectorizer.get_feature_names_out())}")
```


### Step 1: How a Tree Splits
At each node, the Decision Tree looks for the word that best separates "Positive" from "Negative" reviews. It measures this "separation quality" using:
1. **Gini Impurity**: How much "mixed" the labels are in a node (0 means all labels are the same).
2. **Information Gain**: How much the Gini Impurity decreases after a split.



To prevent the tree from becoming infinitely deep and overfitting (memorizing specific reviews), we use **Pruning** by setting a `max_depth`.


```python
# 1. Train Decision Tree
# We limit depth to 5 for interpretability and to prevent overfitting
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42, criterion='gini')
tree_model.fit(X_train, y_train)

# 2. Evaluate
y_pred = tree_model.predict(X_test)
print(f"Decision Tree (Depth=5) Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```


### Step 2: Visualizing the Logic
By plotting the tree, we can follow the exact path the model takes to classify a review. This is the ultimate "White Box" model.


```python
plt.figure(figsize=(25, 12))

# Plot the tree structure
plot_tree(tree_model, 
          feature_names=vectorizer.get_feature_names_out(), 
          class_names=['Negative', 'Positive'], 
          filled=True, 
          rounded=True, 
          fontsize=10)

plt.title("Decision Tree Structure for Sentiment Analysis (Depth=5)")
plt.show()
```


### Step 3: Feature Importance
Even without looking at the whole tree, we can calculate which words (features) contributed the most to the total reduction in impurity across all splits.


```python
# Get feature importances
importances = tree_model.feature_importances_
feature_names = vectorizer.get_feature_names_out()

# Sort and get top 15
indices = np.argsort(importances)[-15:]

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], color='tab:green', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Gini Importance')
plt.title('Top 15 Most Discriminative Words in Decision Tree')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
```

