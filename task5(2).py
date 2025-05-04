import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the data
df = pd.read_csv('heart.csv')
print("✅ Dataset loaded:")
print(df.head())

# Step 2: Split features and labels
X = df.drop('target', axis=1)
y = df['target']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\n📊 Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# Step 5: Visualize Decision Tree
plt.figure(figsize=(16, 10))
plot_tree(dt, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("🌳 Decision Tree Visualization")
plt.show()

# Step 6: Limit Tree Depth (to control overfitting)
dt_limited = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_limited.fit(X_train, y_train)
print("\n📉 Limited Tree Accuracy:", accuracy_score(y_test, dt_limited.predict(X_test)))

# Step 7: Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n🌲 Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Step 8: Feature Importance
feat_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n📌 Feature Importances:\n", feat_importance)

plt.figure(figsize=(10, 5))
sns.barplot(x=feat_importance.values, y=feat_importance.index)
plt.title("🔥 Random Forest Feature Importance")
plt.show()

# Step 9: Cross-validation Accuracy
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"\n🔁 Cross-Validation Accuracy: {cv_scores.mean():.2f}")

input("✅ Press Enter to exit...")
