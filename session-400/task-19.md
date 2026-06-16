# Task 19: XGBoost for Classification


```python
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Standalone setup
DATA_DIR = Path.cwd() / "datasets"

# Load the balanced IMDB dataset
df = pd.read_csv(DATA_DIR / "imdb_balanced_10k.csv")

# Vectorize (Using TF-IDF with 2500 features for fast training)
vectorizer = TfidfVectorizer(max_features=2500, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split data (We need a validation set for tracking learning curves)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} samples.")
```


### Step 1: The Concept of Boosting
Unlike Random Forest (Bagging) which builds many independent deep trees in parallel, **Boosting** builds shallow trees **sequentially**. 

Tree 1 makes predictions. Tree 2 is then trained to predict the **errors (residuals)** of Tree 1. Tree 3 corrects Tree 2, and so on. We are literally applying Gradient Descent, but instead of updating weights, we are adding new trees to step towards the minimum loss.


```python
# 1. Train a base XGBoost model
# use_label_encoder is deprecated, we set it to False to avoid warnings
base_xgb = xgb.XGBClassifier(
    n_estimators=100, 
    random_state=42, 
    use_label_encoder=False, 
    eval_metric='logloss'
)

base_xgb.fit(X_train, y_train)

# Evaluate
base_acc = accuracy_score(y_test, base_xgb.predict(X_test))
print(f"Base XGBoost Accuracy: {base_acc:.4f}")
```


### Step 2: Hyperparameters & Tracking
Gradient Boosting is powerful but prone to overfitting. We control this via:
1. `max_depth`: Keep trees shallow (e.g., 3-6) so they act as "weak learners".
2. `learning_rate` ($\eta$): Scales the contribution of each new tree. A smaller rate means we need more trees, but the learning is more robust.

To see how the model improves over time, we pass an `eval_set` to track the Logloss after every single tree is added.


```python
# 2. Train with specific parameters and track performance
tuned_xgb = xgb.XGBClassifier(
    n_estimators=200,      # Number of boosting rounds (trees)
    learning_rate=0.1,     # Step size shrinkage to prevent overfitting
    max_depth=4,           # Shallow trees
    subsample=0.8,         # Use 80% of data per tree (adds randomness)
    colsample_bytree=0.8,  # Use 80% of features per tree
    random_state=42,
    use_label_encoder=False
)

# We provide both train and test sets to monitor the learning curve
eval_set = [(X_train, y_train), (X_test, y_test)]

# Fit the model and capture the evaluation results
tuned_xgb.fit(
    X_train, y_train, 
    eval_set=eval_set, 
    eval_metric="logloss", 
    verbose=False  # Set to True to see step-by-step logs
)

tuned_acc = accuracy_score(y_test, tuned_xgb.predict(X_test))
print(f"Tuned XGBoost Accuracy: {tuned_acc:.4f}")
```


### Step 3: Visualizing the Learning Curve
By plotting the Logloss of the training and validation sets at each boosting round, we can visualize the Gradient Descent process in action. We look for the point where validation loss stops decreasing—that's where we should stop adding trees.


```python
# Retrieve performance metrics
results = tuned_xgb.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

# Plot Logloss
plt.figure(figsize=(10, 5))
plt.plot(x_axis, results['validation_0']['logloss'], label='Train Logloss', color='tab:blue', linewidth=2)
plt.plot(x_axis, results['validation_1']['logloss'], label='Test Logloss', color='tab:red', linewidth=2)

plt.title('XGBoost Learning Curve (Logloss vs. Boosting Rounds)')
plt.xlabel('Number of Trees (Boosting Rounds)')
plt.ylabel('Logloss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
```

