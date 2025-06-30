import pickle
import os
import sys
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier_copy import TFNaiveBayesClassifier as Classifier
from src.gmail_api import fetch_spam_and_ham

def train_NB_classifier():
    
    if os.path.exists('token.json'):
        os.remove('token.json')
        spam_mails, ham_mails = fetch_spam_and_ham(100)
    else:
        spam_mails, ham_mails = fetch_spam_and_ham(100)
        
    print("Imported spam and ham emails")
        
    spam_mails = {msg_id: metadata for msg_id, metadata in spam_mails.items() if metadata['content'] != ''}
    ham_mails = {msg_id: metadata for msg_id, metadata in ham_mails.items() if metadata['content'] != ''}
    
    print("Filtered out empty emails....")
    
    classifier = Classifier()
    print("Training Naive Bayes classifier")
    classifier.fit(spam_emails=spam_mails, ham_emails=ham_mails)
    
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')
    os.chdir('trained_models')
    with open(f'TF_NaiveBayes_Classifier_{date}.pkl',mode='wb') as file:
        pickle.dump(classifier,file)

    print(f"Training completed and model saved as 'TF_NaiveBayes_Classifier_{date}.pkl'.")

if __name__ == "__main__":
    train_NB_classifier()