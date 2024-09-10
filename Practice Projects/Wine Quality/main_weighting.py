# Add class weights to RandomForestClassifier
clf_weighted = RandomForestClassifier(random_state=42, class_weight='balanced')
clf_weighted.fit(X_train_scaled, y_train_class)

# Predict using the classifier
y_pred_weighted = clf_weighted.predict(X_test_scaled)

# Compute accuracy
accuracy_weighted = accuracy_score(y_test_class, y_pred_weighted)
print(f'Class Weighted Classification Accuracy: {accuracy_weighted}')
