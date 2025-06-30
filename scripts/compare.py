import os
import sys
#make sure to include src module in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classifier_copy import TFNaiveBayesClassifier as Classifier
from src.gmail_api import fetch_spam_and_ham
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def get_emails():
    if os.path.exists('token.json'):
        os.remove('token.json')
        spam_mails, ham_mails = fetch_spam_and_ham(100)
    else:
        spam_mails, ham_mails = fetch_spam_and_ham(100)
    
    spam_mails = {msg_id: metadata for msg_id, metadata in spam_mails.items() if metadata['content'] != ''}
    ham_mails = {msg_id: metadata for msg_id, metadata in ham_mails.items() if metadata['content'] != ''}
    
    return spam_mails, ham_mails

# Fetch emails
spam_mails, ham_mails = get_emails()

X_naive_train, X_naive_test, y_naive_train, y_naive_test = train_test_split(spam_mails.update(ham_mails),['spam'] * len(spam_mails) + ['ham'] * len(ham_mails), test_size=0.2, random_state=42)

X = list(spam_mails.values()) + list(ham_mails.values())
y = ['spam'] * len(spam_mails) + ['ham'] * len(ham_mails)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = Classifier()
classifier.fit(X_train, y_train)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
sklearn_classifier = MultinomialNB()
sklearn_classifier.fit(X_train_vectorized, y_train)

# Compare the two classifiers
train_accuracy = classifier.score(X_train, y_train)
test_accuracy = classifier.score(X_test, y_test)
sklearn_train_accuracy = sklearn_classifier.score(X_train_vectorized, y_train)
sklearn_test_accuracy = sklearn_classifier.score(X_test_vectorized, y_test)

print(f"Custom Classifier Train Accuracy: {train_accuracy}")
print(f"Custom Classifier Test Accuracy: {test_accuracy}")
print(f"Sklearn Classifier Train Accuracy: {sklearn_train_accuracy}")
print(f"Sklearn Classifier Test Accuracy: {sklearn_test_accuracy}")
