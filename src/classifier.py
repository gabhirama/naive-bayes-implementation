import math

class TFNaiveBayesClassifier:
    def __init__(self, spam_word_count, ham_word_count, p_spam, p_ham, total_words_in_spam, total_words_in_ham, vocab_size, laplace_smoothing_factor):
        self.spam_word_count = spam_word_count
        self.ham_word_count = ham_word_count
        self.p_spam = p_spam
        self.p_ham = p_ham
        self.total_words_in_spam = total_words_in_spam
        self.total_words_in_ham = total_words_in_ham
        self.vocab_size = vocab_size
        self.laplace_smoothing_factor = laplace_smoothing_factor
        
    def update_classifier(self, spam_emails, ham_emails):
        """Update the classifier with new spam and ham emails"""
        # Update word counts
        for email in spam_emails:
            for word in email.lower().split():
                self.spam_word_count[word] = self.spam_word_count.get(word, 0) + 1
                self.total_words_in_spam += 1
        
        for email in ham_emails:
            for word in email.lower().split():
                self.ham_word_count[word] = self.ham_word_count.get(word, 0) + 1
                self.total_words_in_ham += 1
        
        # Update vocabulary size
        self.vocab_size = len(set(self.spam_word_count.keys()).union(set(self.ham_word_count.keys())))
        
        # Update probabilities
        total_emails = len(spam_emails) + len(ham_emails)
        self.p_spam = len(spam_emails) / total_emails if total_emails > 0 else 0.5
        self.p_ham = len(ham_emails) / total_emails if total_emails > 0 else 0.5
    
    def get_word_probability(self,word,class_word_count_dict,total_words_in_class):
        """Calculate the P(word|Class) with laplace smoothing"""
        word_count = class_word_count_dict.get(word, 0)
        probability = (word_count + self.laplace_smoothing_factor)/(total_words_in_class + self.laplace_smoothing_factor*self.vocab_size)
        return probability
    
    def get_word_probability_total(self, word):
        """Calculate the P(word) occuring in the training set (with laplace smoothing)"""
        spam_probability = self.get_word_probability(word, self.spam_word_count, self.total_words_in_spam)
        ham_probability = self.get_word_probability(word, self.ham_word_count, self.total_words_in_ham)
        return spam_probability*self.p_spam + ham_probability*self.p_ham
    
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