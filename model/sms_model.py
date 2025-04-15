import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
data = pd.read_csv("../data/spam.csv", encoding="latin1", skiprows=1, names=["label", "text"])

# 🛠 Drop missing text values
data = data.dropna(subset=["text"])


# 2. Convert 'ham' and 'spam' to 0 and 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42)

# 4. Convert text into TF-IDF features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {accuracy:.2f}")

# 7. Save the model and vectorizer
joblib.dump(model, "../pkl/spam_classifier_model.pkl")
joblib.dump(vectorizer, "../pkl/tfidf_vectorizer.pkl")
print("✅ Model and vectorizer saved as .pkl files!")

# 8. Function to predict new message
def predict_message(msg):
    msg_vec = vectorizer.transform([msg])
    pred = model.predict(msg_vec)[0]
    return "SPAM" if pred == 1 else "NOT SPAM"

# Test the function
#print(predict_message("Free prize awaits! Claim now"))
#print(predict_message("Hey, what’s the plan for tomorrow?"))
