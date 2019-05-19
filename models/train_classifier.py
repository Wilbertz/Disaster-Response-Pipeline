"""train_classifier: Python methods for training a classifier."""

__author__ = "Harald Wilbertz"
__version__ = "1.0.0"

import sys
import re
import pickle
import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, precision_recall_fscore_support
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def load_data(database_filepath: str) -> Tuple[pd.Series, pd.DataFrame, List[str]]:
    """
       Load the data from the message table within the SQLite database
       and transform them to data structures suitable for machine
       learning algorithms.

       arguments:
           database_filepath (string): The full path to the SQLite database file.
       returns:
           A tuple consisting of
           - The input values as a pandas series. Each individual input value is a string.
           - The output values as a pandas data frame.
           - The column names of the output values as a list of strings.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    x = df.loc[:, 'message']
    y = df.iloc[:, -36:]
    category_names = list(y.columns)

    return x, y, category_names


def tokenize(text: str) -> List[str]:
    """
       Tokenize the given text, this includes the following steps:
       - punctuation removal
       - tokenize words
       - stopwords removal
       - lemmatize words

       arguments:
           text (string): input text to be tokenized.
       returns:
           tokens (list of strings) : list of tokens
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).strip()

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]

    return tokens


def multiclass_f1_score(y_test: np.ndarray, y_pred: np.ndarray) -> np.float:
    """
        Computes the F1 score (harmonic mean of precision and recall). The result is
        the weighted average of the f1 scores for individual categories.

        arguments:
            y_test (ndarray): true values
            y_pred (ndarray): predicted values
        returns:
            f1_score (float): The weighted average of f1 scores.
    """
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    f1_scores = []
    for i, _ in enumerate(range(y_test.shape[1])):
        f1_scores.append(f1_score(y_true=y_test[:, i], y_pred=y_pred[:, i], average='weighted'))

    result = np.average(f1_scores)
    logging.info('Multiclass F1 Score {}'.format(result))

    return result


def build_model() -> GridSearchCV:
    """
       Build a GridSearchCV object that contains machine learning model with a pipeline
       consisting of the following steps:

        - CountVectorizer
        - A TFIDF (term frequency–inverse document frequency) Transformer
        - A RandomForest based Multioutput Classifier

       Attention ! : There exists a bug within joblib that results in the error message
       "ValueError: UPDATEIFCOPY base is read-only" in case n_jobs is set to a value of -1.
       In order to avoid this error, I had to use a value of 1, forcing the program to use just
       one processor. For more details see: https://github.com/scikit-learn/scikit-learn/issues/6614

       arguments:
           None
       returns:
            A GridSearchCV object ready to use

    """
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=1))
    ])
    parameters = {
        'vectorizer__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': (10, 100)
    }
    # Create a f1 scorer
    scorer = make_scorer(multiclass_f1_score, greater_is_better=True)

    return GridSearchCV(pipeline, param_grid=parameters, verbose=1, refit=True, n_jobs=1, scoring=scorer)


def evaluate_model(model: GridSearchCV, x_test: pd.Series, y_test: pd.DataFrame, category_names: List[str]) -> None:
    """
       This method evaluates a trained model by computing the following scores for the different categories:
        - Precision
        - Recall
        - F1 Score
       The result are printed on the console.

       arguments:
            model (GridSearchCV): The machine learning model to be evaluated
            x_test (Series): The input values
            y_test (DataFrame): The output values
            category_name (List of strings): The names of the categories
       returns:
           None
    """
    y_pred = model.predict(x_test)
    log_and_print_evaluation_report(y_test, y_pred, category_names)


def log_and_print_evaluation_report(y_test: pd.DataFrame, y_pred: np.ndarray, category_names: List[str]) -> None:
    """
        Compute precision, recall, f1-score for all categories

        arguments:
            y_test (numpy array): result of prediction
            y_pred (numpy array): true values
            category_names (list of string): List with category names
    """
    precisions, recalls, f1_scores = [], [], []

    y_test, y_pred = np.array(y_test), np.array(y_pred)

    for i, _ in enumerate(range(y_test.shape[1])):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test[:, i], y_pred[:, i], average='weighted', warn_for=tuple())
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        log_and_print('Category = {}, Precision =  {}, Recall = {}, F1 Score = {}'.format(
              category_names[i], precision, recall, f1))

    log_and_print('\nFINAL\nPrecision = {}, Recall = {}, F1 Score = {}'.format(
          np.average(precisions), np.average(recalls), np.average(f1_scores)))


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    """
       Save the machine learning model as a pickle dump.

       arguments:
            model (GridSearchCV): The machine learning model to be saved
            model_filepath (string): The full path to the pickle file
       returns:
           None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def log_and_print(log_message: str) -> None:
    """
        This method outputs its given string on the console and
        as a log message with info log level.

        arguments:
           log_message (string): The string to be logged.
       returns:
           None
    """
    print(log_message)
    logging.info(log_message)


def main() -> None:
    """
       This is the main entry point into the train classifier program. Running
       the program creates either a new classifier.pkl file or overwrites an existing file.
       The following sequence of actions is executed:
        - Load data from database.
        - Build a machine learning model.
        - Train the model.
        - Evaluate the model.
        - Persist the model as a pickle file.

       command line arguments:
            database_filepath (String): The full path to the SQLLite database.
            model_filepath (string): The full path to the pickle file for the model.
       returns:
           None
    """
    logging.basicConfig(
        filename='train_classifier.log',
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO)
    logging.info('Started with arguments: {}'.format(sys.argv))

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        log_and_print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        x, y, category_names = load_data(database_filepath)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        log_and_print('Building model...')
        model = build_model()

        log_and_print('Training model...')
        model.fit(x_train, y_train)

        log_and_print('Evaluating model...')
        evaluate_model(model, x_test, y_test, category_names)

        log_and_print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        log_and_print('Trained model saved!')

    else:
        log_and_print('Please provide the filepath of the disaster messages database '
                      'as the first argument and the filepath of the pickle file to '
                      'save the model to as the second argument. \n\nExample: python '
                      'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
