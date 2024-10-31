import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

TEST_SIZE = 0.2
RANDOM_STATE = 7
FILE_PATH = '../../Practice Projects/Datasets/pima-indians-diabetes.csv'

data = pd.read_csv(FILE_PATH, header=None)
data.columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

# Replace zero values with NaN for specific columns
columns_with_zero_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_with_zero_values] = data[columns_with_zero_values].replace(0, pd.NA)

# Fill missing values with the mean of the column and convert the columns to numeric
data.fillna(data.mean(), inplace=True)
data.infer_objects(copy=False)

# Split the data into features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and train the model
model = RandomForestClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)
