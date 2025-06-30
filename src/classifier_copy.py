import math
from collections import Counter

class TFNaiveBayesClassifier:
    def __init__(self, laplace_smoothing_factor=1):
        self.laplace_smoothing_factor = laplace_smoothing_factor
        if self.laplace_smoothing_factor <= 0:
            raise ValueError("Laplace smoothing factor must be greater than 0")
        
        self.spam_word_count = Counter()
        self.ham_word_count = Counter()
        self.total_words_in_spam = 0
        self.total_words_in_ham = 0
        self.p_spam = 0.0
        self.p_ham = 0.0
        self.vocab = set()
        self.vocab_size = 0
        self.is_fitted = False
        
    def fit(self,emails, labels):
        # Edge cases where .fit() method might fail due to invalid inputs
        if not isinstance(emails, list) or not isinstance(labels, list):
            raise ValueError("Emails and labels must be lists")
        if len(emails) == 0 or len(labels) == 0:
            raise ValueError("Both emails and labels must contain at least one email for training")
        if len(emails) != len(labels):
            raise ValueError("Emails and labels must have the same length")
        if self.is_fitted:
            raise RuntimeError("Classifier is already fitted. Create a new instance to fit again")
        
        # Uncomment the following lines if you want to reset the classifier before fitting
        # self.spam_word_count.clear()
        # self.ham_word_count.clear()
        # self.total_words_in_spam = 0
        # self.total_words_in_ham = 0
        # self.vocab.clear()
        
        spam_emails = {email: {'content': emails[i]} for i, email in enumerate(emails) if labels[i] == 'spam'}
        ham_emails = {email: {'content': emails[i]} for i, email in enumerate(emails) if labels[i] == 'ham'}
        
        for email in list(spam_emails.values()):
            words = email['content'].split()
            words = [word.lower() for word in words]  # Convert words to lowercase
            words = [word.strip('.,!?;:"()[]}{') for word in words]
            for word in words:
                self.spam_word_count[word] += 1
                self.total_words_in_spam += 1
                self.vocab.add(word)

        for email in list(ham_emails.values()):
            words = email['content'].split()
            words = [word.lower() for word in words]  # Convert words to lowercase
            words = [word.strip('.,!?;:"()[]}{') for word in words]
            for word in words:
                self.ham_word_count[word] += 1
                self.total_words_in_ham += 1
                self.vocab.add(word)

        self.vocab_size = len(self.vocab)
        self.p_spam = (len(spam_emails) + self.laplace_smoothing_factor) / (len(spam_emails) + len(ham_emails) + self.laplace_smoothing_factor * 2)
        self.p_ham = (len(ham_emails) + self.laplace_smoothing_factor) / (len(spam_emails) + len(ham_emails) + self.laplace_smoothing_factor * 2)

    def get_word_probability(self,word,class_word_count_dict,total_words_in_class):
        """Calculate the P(word|Class) with laplace smoothing"""
        word_count = class_word_count_dict.get(word, 0)
        probability = (word_count + self.laplace_smoothing_factor)/(total_words_in_class + self.laplace_smoothing_factor*self.vocab_size)
        return probability
    
    def get_word_probability_total(self, word):
        """Calculate the P(word) occuring in the training set = P(word|spam) * P(spam) + P(word|ham) * P(ham)"""
        spam_probability = self.get_word_probability(word, self.spam_word_count, self.total_words_in_spam)
        ham_probability = self.get_word_probability(word, self.ham_word_count, self.total_words_in_ham)
        total_probability = spam_probability * self.p_spam + ham_probability * self.p_ham
        return total_probability
    
    def classification_probability(self, email_content):
        """Classify the email content as spam or ham"""
        words = email_content.lower().split()
        
        # probability_spam_given_words = self.p_spam
        # probability_ham_given_words = self.p_ham
        
        log_probability_spam = math.log(self.p_spam)
        log_probability_ham = math.log(self.p_ham)
        
        for word in words:
            probability_word_given_spam = self.get_word_probability(word,self.spam_word_count,self.total_words_in_spam)
            probability_word_given_ham = self.get_word_probability(word,self.ham_word_count,self.total_words_in_ham)
            
            # probability_spam_given_words *= probability_word_given_spam
            # probability_ham_given_words *= probability_word_given_ham
            
            log_probability_spam += math.log(probability_word_given_spam)
            log_probability_ham += math.log(probability_word_given_ham)
            
        # Normalize the probabilities
        # total_probability = probability_spam_given_words + probability_ham_given_words
        max_log_probability = max(log_probability_spam, log_probability_ham)
        
        log_probability_spam -= max_log_probability
        log_probability_ham -= max_log_probability
        probability_spam_given_words = math.exp(log_probability_spam)
        probability_ham_given_words = math.exp(log_probability_ham)
        
        total_probability = probability_spam_given_words + probability_ham_given_words
        
        #Final probabilities
        probability_spam_given_words /= total_probability
        probability_ham_given_words /= total_probability
        
        self.is_fitted = True
        
        return {
            'spam_probability': probability_spam_given_words,
            'ham_probability': probability_ham_given_words
        }
        
    def classify(self, email_content):
        """Classify the email content as spam or ham"""
        probabilities = self.classification_probability(email_content)
        if probabilities['spam_probability'] > probabilities['ham_probability']:
            return 'spam'
        else:
            return 'ham'
        
    def score(self, emails,labels):
        """Calculate the accuracy of the classifier on the given emails and labels"""
        if not self.is_fitted:
            raise RuntimeError("Classifier is not fitted yet. Call fit() method before scoring.")
        
        if len(emails) != len(labels):
            raise ValueError("Number of emails and labels must be the same")
        
        correct_predictions = 0
        for email, label in zip(emails, labels):
            predicted_label = self.classify(email)
            if predicted_label == label:
                correct_predictions += 1
        
        return correct_predictions / len(emails)
        
    def save_model(self, file_path):
        """Save the trained model to a file"""
        import pickle
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
            
    @staticmethod
    def load_model(file_path):
        """Load the trained model from a file"""
        import pickle
        with open(file_path, 'rb') as file:
            return pickle.load(file)