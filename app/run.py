import json
import re
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly
# import as pgo and not go in order to avoid name clash with go controller method
import plotly.graph_objs as pgo
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text: str):
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

# load model
model = joblib.load("../models/classifier.pkl")


# index web page displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    df_top_10_cat = df.iloc[:, 4:].sum(axis=0).sort_values(ascending=False)[:10]
    top_10_cat_names = list(df_top_10_cat.index.values)
    top_10_cat_values = list(df_top_10_cat.values)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
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
        }
    ]
    graph_2 = {
        'data': [
            pgo.Bar(
                x=top_10_cat_names,
                y=top_10_cat_values
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
    graphs.append(graph_2)
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


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