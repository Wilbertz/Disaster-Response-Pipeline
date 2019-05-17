"""train_classifier: Python methods for training a classifier."""

__author__ = "Harald Wilbertz"
__version__ = "1.0.0"

import sys
import re
import pickle
from typing import List, Tuple
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
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

    
def build_model() -> GridSearchCV:
    """
       Build a GridSearchCV object that contains machine learning model with a pipeline
       consisting of the following steps:

        - CountVectorizer
        - A TFIDF (term frequency–inverse document frequency) Transformer
        - A RandomForest based Multioutput Classifier

       arguments:
           None
       returns:
            A GridSearchCV object ready to use

    """
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'tfidf__use_idf': [True],
    }
    return GridSearchCV(pipeline, param_grid=parameters, verbose=2)


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
    print(classification_report(y_test, y_pred, target_names=category_names))


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
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        x, y, category_names = load_data(database_filepath)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(x_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, x_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
