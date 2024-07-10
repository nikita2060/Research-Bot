# README for Spam Email Detection Using Naive Bayes

## Overview
This project demonstrates the implementation of a spam email detection model using the Naive Bayes algorithm. The goal is to classify emails as either spam or ham (non-spam) based on their content.

## Prerequisites
Ensure you have the following libraries installed:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

Install the required packages using:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Dataset
The dataset `emails.csv` contains:
- `text`: The email content.
- `spam`: Label indicating whether the email is spam (1) or ham (0).

## Steps

### 1. Load Libraries and Dataset
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

spam_df = pd.read_csv('emails.csv')
```

### 2. Data Exploration
View the dataset and its structure:
```python
spam_df.head(10)
spam_df.tail(10)
spam_df.describe()
spam_df.info()
```

### 3. Visualize Data
```python
sns.countplot(spam_df['spam'], label='Spam vs Ham')
plt.show()
```

### 4. Text Vectorization
Convert text data to numerical data using CountVectorizer:
```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
spamham_countVectorizer = vectorizer.fit_transform(spam_df['text'])
```

### 5. Split Data into Training and Testing Sets
```python
from sklearn.model_selection import train_test_split
X = spamham_countVectorizer
y = spam_df['spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### 6. Train Naive Bayes Model
```python
from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)
```

### 7. Evaluate the Model
Evaluate using confusion matrix and classification report:
```python
from sklearn.metrics import classification_report, confusion_matrix
y_predict_test = NB_classifier.predict(X_test)

cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test, y_predict_test))
```

## Results
The model achieves high accuracy in detecting spam emails, as indicated by the classification report and confusion matrix.

## Conclusion
This project successfully demonstrates how to build and evaluate a spam detection model using Naive Bayes, achieving high precision and recall in distinguishing between spam and ham emails.
