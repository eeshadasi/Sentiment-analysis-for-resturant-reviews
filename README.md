# Sentiment Analysis on Restaurant Reviews

## Overview
This project performs sentiment analysis on restaurant reviews using Natural Language Processing (NLP) techniques. It preprocesses the text data, converts it into numerical form, and applies machine learning models such as Random Forest and Naïve Bayes for classification.

## Dataset
- The dataset used is `Restaurant_Reviews.tsv`, which contains restaurant reviews and their corresponding sentiment labels.
- It is a tab-separated values (TSV) file with two columns:
  - **Review**: Text review of the restaurant.
  - **Sentiment**: Binary label (0 for negative, 1 for positive).

## Dependencies
Ensure you have the following Python libraries installed before running the script:
```bash
pip install numpy pandas nltk scikit-learn
```

## Steps Performed
### 1. Import Required Libraries
The script uses `numpy`, `pandas`, and `nltk` for data processing and `sklearn` for machine learning tasks.

### 2. Data Preprocessing
- Tokenizes text using NLTK's `word_tokenize`.
- Removes non-alphabetic characters.
- Converts text to lowercase.
- Removes stopwords.
- Applies stemming using the PorterStemmer.
- Creates a corpus of cleaned reviews.

### 3. Feature Extraction
- Converts text into a bag-of-words representation using `CountVectorizer` (with `max_features=1500`).
- Converts the transformed text into a NumPy array.

### 4. Train-Test Split
- Splits the dataset into 70% training and 30% testing data using `train_test_split()`.

### 5. Model Training and Evaluation
#### **Random Forest Classifier**
- Performs hyperparameter tuning using `GridSearchCV` to find the best parameters.
- Trains a Random Forest model using the best parameters.
- Evaluates the model using Accuracy, Precision, and Recall metrics.

#### **Naïve Bayes Classifier**
- Trains a `MultinomialNB` classifier with `alpha=0.1`.
- Evaluates the model using Accuracy, Precision, and Recall metrics.

## Results
- The accuracy, precision, and recall for both classifiers are printed after model evaluation.

## Usage
1. Place `Restaurant_Reviews.tsv` in the working directory.
2. Run the script using:
   ```bash
   python script.py
   ```
3. View the output for model performance metrics.

## Future Improvements
- Experiment with different text vectorization techniques like TF-IDF.
- Try deep learning models such as LSTMs or Transformers for improved performance.
- Perform sentiment analysis on a larger dataset for better generalization.

## License
This project is for educational purposes only. Feel free to modify and experiment with the code.

