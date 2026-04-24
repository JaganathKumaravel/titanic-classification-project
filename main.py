# =========================
# 1. IMPORTS
# =========================
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =========================
# 2. LOAD DATA
# =========================
def load_data():
    return pd.read_csv('Titanic-Dataset.csv')   # <-- Put dataset in same folder

df = load_data()


# =========================
# 3. PREPROCESSING
# =========================
def preprocess(df):
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['family_size'] = df['SibSp'] + df['Parch']
    return df

df = preprocess(df)


# =========================
# 4. SPLIT DATA
# =========================
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 5. PIPELINE (IMPORTANT)
# =========================
categorical_cols = ['Sex', 'Embarked']
numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'family_size']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ]
)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])


# =========================
# 6. TRAIN MODEL
# =========================
pipeline.fit(X_train, y_train)


# =========================
# 7. EVALUATION
# =========================
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# =========================
# 8. SAVE MODEL
# =========================
joblib.dump(pipeline, 'model.pkl')

print("\n Model saved as model.pkl")




    




        
    












        