"""run.py: Flask based web application."""

__author__ = "Harald Wilbertz"
__version__ = "1.0.0"

import re
import json
from typing import List
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request
import plotly
# import as pgo and not go in order to avoid name clash with go controller method
import plotly.graph_objs as pgo
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.sqlite')
df = pd.read_sql_table('Messages', engine)

# load model from pickl file
model = joblib.load("../models/classifier.pkl")


# index web page displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    top_categories = df.iloc[:, 4:].sum(axis=0).sort_values(ascending=False)[:10]
    top_category_names = list(top_categories.index.values)
    top_category_values = list(top_categories.values)

    # create visuals
    graphs = [
        {
            'data': [
                pgo.Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                pgo.Bar(
                    x=top_category_names,
                    y=top_category_values
                )
            ],

            'layout': {
                'title': 'Top 10 categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graph_json)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
