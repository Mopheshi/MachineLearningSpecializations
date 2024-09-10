from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Apply SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train_class)

# Train RandomForest on the resampled data
clf_smote = RandomForestClassifier(random_state=42)
clf_smote.fit(X_resampled, y_resampled)

# Predict using the classifier
y_pred_smote = clf_smote.predict(X_test_scaled)

# Compute accuracy
accuracy_smote = accuracy_score(y_test_class, y_pred_smote)
print(f'SMOTE Classification Accuracy: {accuracy_smote}')
