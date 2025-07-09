import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score,f1_score, classification_report, recall_score
from imblearn.combine import SMOTETomek
from collections import Counter
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Load and preprocess the data
df = pd.read_csv('spam.csv', encoding='cp1252')
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)
df['target'] = df['target'].map({'ham': 0, 'spam': 1})
df = df.drop_duplicates()
df['transformed_text'] = df['text'].apply(transform_text)

# Create TF-IDF vectors
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining set distribution before SMOTE:")
print(Counter(y_train))
# Apply SMOTETomek to balance the data
smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
print("\nTraining set distribution after SMOTETomek:")
print(Counter(y_train_balanced))

# Create and train the model
mnb = MultinomialNB()
mnb.fit(X_train_balanced, y_train_balanced)
# Make predictions
y_train_pred = mnb.predict(X_train_balanced)
y_test_pred = mnb.predict(X_test)

print("\n Metrics:")
print(f"Training Accuracy: {accuracy_score(y_train_balanced, y_train_pred):.4f}")
print(f"Testing Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Testing Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Testing Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"Testing F1-score: {f1_score(y_test, y_test_pred):.4f}")

def evaluate_model(X_train, X_test, y_train, y_test, use_smote=False):
    if use_smote:
        smote_tomek = SMOTETomek(random_state=42)
        X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train_res, y_train_res)
    
    # Make predictions
    y_train_pred = model.predict(X_train_res)
    y_test_pred = model.predict(X_test)
    
    # Print results
    print(f"\nResults {'with' if use_smote else 'without'} SMOTETomek:")
    
    print("\nTraining Metrics:")
    print(f"Accuracy: {accuracy_score(y_train_res, y_train_pred):.4f}")
    print(f"Precision: {precision_score(y_train_res, y_train_pred):.4f}")
    print(f"Recall: {recall_score(y_train_res, y_train_pred):.4f}")
    print(f"F1-score: {f1_score(y_train_res, y_train_pred):.4f}")
    
    print("\nTest Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_test_pred):.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    return model

# Compare both approaches
print("Model Performance Comparison:")
model_without_smote = evaluate_model(X_train, X_test, y_train, y_test, use_smote=False)
model_with_smote = evaluate_model(X_train, X_test, y_train, y_test, use_smote=True)

# Save the trained model and vectorizer
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))