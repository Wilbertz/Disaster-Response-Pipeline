# Disaster Response Pipeline
Python code for a disaster response pipeline.

## Table of Contents

1. [Installation](#installation)
2. [Instructions](#instructions)
3. [Directory Structure](#directoryStructure)
4. [Design](#design)
5. [Results](#results)

## Installation <a name="installation"></a>

This project was written in Python 3.7 using Python 3 type hints. The relevant Python packages for this project are as follows:
- re
- os  
- sys  
- json
- logging 
- typing 
- unittest
- pickle  
- pandas  
- numpy  
- nltk.stem  
- nltk.tokenize
- nltk.corpus
- flask
- plotly  
- sklearn.externals
- sklearn.metrics  
- sklearn.pipeline 
- sklearn.feature_extraction.text  
- sklearn.multioutput  
- sklearn.ensemble  
- sklearn.model_selection  
- sqlalchemy

The nltk.download method is called with 'punkt', 'wordnet', and 'stopwords'.

## Instructions <a name="instructions"></a>
To run ETL pipeline that does all the cleaning of the data and stores the result in SQLite database:

`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.sqlite`

To run the machine learning  pipeline that trains and saves the classifier:

`python models/train_classifier.py data/DisasterResponse.sqlite models/classifier.pkl`

__Attention ! This command takes 5 - 6 hours to complete.__

Run the following command in the app subdirectory to run the web application: 

`python run.py`

Open http://0.0.0.0:3001/ (or http://localhost:3001/ depending on your operation system) in your browser.

## Directory Structure <a name="directoryStructure"></a>

- Root /

    - README.md  
    - app /
        - templates /
            - go.html
            - master.html 
        - run.py  
    - data /  
        - disaster_categories.csv  
        - disaster_messages.csv  
        - process_data.py  
    - tests /  
        - all_tests.py  
        - process_data_test.py  
        - train_classifier_test.py  
        - unittest_disaster_categories.csv  
        - unittest_disaster_messages.csv  
    - models /  
        - train_classifier.py  
        - classifier.pkl  
    - uml /
        - Use cases.png  
        - Components.png

## Design <a name="design"></a>

### Main Uses Cases
There are 3 main use cases:

- Import data
- Train model
- Classify message

<p align="center">
    <img src="./uml/Use cases.png" width="800" title="Main use cases." alt="Main use cases.">
</p>

### Components and Dataflow

There are 3 main components:

- process_data.py  
- train_classifier.py  
- run.py  

<p align="center">
    <img src="./uml/Components.png" width="800" title="Components and Dataflow." alt="Components and Dataflow.">
</p>

#### process_data.py
Running process_data.py loads data from the messages and categories csv file and
persists this data into a SQLite database. Any existing database file
will be overwritten. Log messages are written both to the console and
a dedicated log file named process_data.log.

The following sequence of actions is executed:
- Load data csv files  
- Clean the data  
- Save the data into a SQLite database  

#### train_classifier.py

Running train_classifier.py creates either a new classifier.pkl file or overwrites an existing file.

The following sequence of actions is executed:
- Load data from database  
- Build a machine learning model  
- Train the model  
- Evaluate the model   
- Persist the model as a pickle file  

#### run.py

This is a web application using the Flask framework. During startup the application loads both the machine learning model from the pickle file and messages from the SQLite database.

## Results <a name="results"></a>

The trained classifier achieves the following classification results:

- Precision = 0.939
- Recall = 0.949  
- F1 Score = 0.937  

For the individual categories:

- related, Precision =  0.812, Recall = 0.824, F1 Score = 0.806  
- request, Precision =  0.887, Recall = 0.891, F1 Score = 0.879  
- offer, Precision =  0.993, Recall = 0.996, F1 Score = 0.995  
- aid_related, Precision =  0.780, Recall = 0.781, F1 Score = 0.779    
- medical_help, Precision =  0.910, Recall = 0.925, F1 Score = 0.900  
- medical_products, Precision =  0.942, Recall = 0.952, F1 Score = 0.934  
- search_and_rescue, Precision =  0.964, Recall = 0.972, F1 Score = 0.961  
- security, Precision =  0.961, Recall = 0.980, F1 Score = 0.970   
- military, Precision =  0.9667, Recall = 0.970, F1 Score = 0.959  
- child_alone, Precision =  1.0, Recall = 1.0, F1 Score = 1.0  
- water, Precision =  0.953, Recall = 0.956, F1 Score = 0.948  
- food, Precision =  0.938, Recall = 0.942, F1 Score = 0.937  
- shelter, Precision =  0.929, Recall = 0.935, F1 Score = 0.924  
- clothing, Precision =  0.983, Recall = 0.985, F1 Score = 0.980  
- money, Precision =  0.968, Recall = 0.976, F1 Score = 0.966  
- missing_people, Precision =  0.978, Recall = 0.989, F1 Score = 0.984  
- refugees, Precision =  0.966, Recall = 0.965, F1 Score = 0.949  
- death, Precision =  0.954, Recall = 0.959, F1 Score = 0.945  
- other_aid, Precision =  0.846, Recall = 0.870, F1 Score = 0.817  
- infrastructure_related, Precision =  0.873, Recall = 0.934, F1 Score = 0.902  
- transport, Precision =  0.943, Recall = 0.956, F1 Score = 0.939  
- buildings, Precision =  0.949, Recall = 0.953, F1 Score = 0.937  
- electricity, Precision =  0.981, Recall = 0.980, F1 Score = 0.972  
- tools, Precision =  0.988, Recall = 0.994, F1 Score = 0.991  
- hospitals, Precision =  0.978, Recall = 0.988, F1 Score = 0.983  
- shops, Precision =  0.991, Recall = 0.995, F1 Score = 0.993  
- aid_centers, Precision =  0.976, Recall = 0.987, F1 Score = 0.982  
- other_infrastructure, Precision =  0.916, Recall = 0.957, F1 Score = 0.936  
- weather_related, Precision =  0.878, Recall = 0.880, F1 Score = 0.876  
- floods, Precision =  0.950, Recall = 0.952, F1 Score = 0.946  
- storm, Precision =  0.938, Recall = 0.943, F1 Score = 0.939  
- fire, Precision =  0.984, Recall = 0.991, F1 Score = 0.988  
- earthquake, Precision =  0.973, Recall = 0.974, F1 Score = 0.973  
- cold, Precision =  0.979, Recall = 0.980, F1 Score = 0.973  
- other_weather, Precision =  0.932, Recall = 0.947, F1 Score = 0.925  
- direct_report, Precision =  0.844, Recall = 0.852, F1 Score = 0.829  
