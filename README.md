# Naive Bayes Email Spam Classifier

This project implements a custom Naive Bayes classifier from scratch in Python to classify emails as spam or ham (not spam). It demonstrates both the theory and practical aspects of building, training, and using a probabilistic classifier, with a focus on understanding the end-to-end workflow from data fetching to model deployment.

---

## Project Structure

```
naive-bayes-implementation/
│
├── src/
│   ├── gmail_api.py              # Functions to fetch spam and ham emails
│   └── classifier_copy.py        # Custom Naive Bayes classifier class
│
├── scripts/
│   ├── train.py                  # Script to train and save the classifier
│   └── classify_email.py         # (To be implemented) Script to load model and classify new emails
│
├── notebooks/
│   └── test.ipynb                # Jupyter notebook for exploration and prototyping
│
├── trained_models/               # Saved trained models (created by train.py)
│
└── README.md                     # Project documentation
```

---

## Key Concepts & Learnings

- **Theory vs. Practice:**  
  While theory focuses on the Naive Bayes algorithm and its assumptions, implementing it in code requires careful handling of data structures, training logic, and numerical stability.

- **Custom Class Implementation:**  
  The classifier is implemented as a Python class with `.fit()` for training and `.classify()` for prediction, encapsulating all logic and making the model reusable.

- **Training (`fit` method):**  
  - Iterates over spam and ham emails.
  - Tokenizes and counts word occurrences per class.
  - Builds a vocabulary set.
  - Calculates

## Future scopes

- **Implemenation of IDF for feature engineering**
- **Comparision with a discriminative model and sklearn's multinomialNB**
