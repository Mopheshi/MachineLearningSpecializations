from main import X_train_scaled, X_test_scaled, y

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Convert the target variable to classification
y_class = y.round().astype(int)

# Split into train and test sets
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train_class)

# Predict using the classifier
y_pred_class = clf.predict(X_test_scaled)

# Compute accuracy
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f'Classification Accuracy: {accuracy}')
