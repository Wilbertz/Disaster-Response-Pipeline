"""train_classifier: Python methods for training a classifier."""

__author__ = "Harald Wilbertz"
__version__ = "1.0.0"

import sys
import re
import pickle

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


def load_data(database_filepath: str):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    x = df.loc[:, 'message']
    y = df.iloc[:, -36:]
    category_names = list(y.columns)

    return x, y, category_names


def tokenize(text: str):
    """
           Tokenize the given text
           - punctuation removal
           - tokenize words
           - stopwords removal
           - lemmatize words

           arguments:
               text (string): input text to tokenize
           returns:
               lemmed_words (list) : list of tokens
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).strip()

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]

    return tokens

    
def build_model():
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {'tfidf__use_idf': [True], }
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    return model


def evaluate_model(model, x_test, y_test, category_names):
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath: str):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
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
