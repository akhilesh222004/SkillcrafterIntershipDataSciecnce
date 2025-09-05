"""
Task 3: Decision Tree Classifier
Dataset: Bank Marketing (UCI ID 222 via ucimlrepo)
Goal: Predict if a customer subscribes to a product (y = yes/no)
"""

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from ucimlrepo import fetch_ucirepo
import graphviz
import os

# --------------------------
# 2. Load dataset
# --------------------------
bank = fetch_ucirepo(id=222)

# Features (X) and target (y)
X = bank.data.features
y = bank.data.targets

print("Dataset shape:", X.shape, y.shape)
print("\nTarget distribution:")
print(y.value_counts())

# --------------------------
# 3. Encode categorical variables
# --------------------------
X_encoded = pd.get_dummies(X, drop_first=True)
print("Shape after encoding:", X_encoded.shape)

# --------------------------
# 4. Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)

# --------------------------
# 5. Train Decision Tree Classifier
# --------------------------
clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,   # limit depth for interpretability
    random_state=42
)
clf.fit(X_train, y_train)

# --------------------------
# 6. Predictions & Evaluation
# --------------------------
y_pred = clf.predict(X_test)

print("\n--- Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------
# 7. Visualization (Matplotlib Preview)
# --------------------------
plt.figure(figsize=(40, 20))  # enlarge plot
plot_tree(
    clf,
    filled=True,
    feature_names=X_encoded.columns,
    class_names=clf.classes_,
    fontsize=8,
    max_depth=3   # show only top 3 levels for readability
)
plt.title("Decision Tree Classifier - Preview (Top 3 Levels)")
plt.show()

# --------------------------
# 8. Export Tree as High-Quality PNG (Graphviz)
# --------------------------
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=X_encoded.columns,
    class_names=clf.classes_,
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)

# Ensure output directory exists
output_dir = "outputs_task3"
os.makedirs(output_dir, exist_ok=True)

# Save PNG
png_path = os.path.join(output_dir, "decision_tree")
graph.render(png_path, format="png", cleanup=True)

print(f"\nDecision tree exported as high-quality PNG: {png_path}.png")
