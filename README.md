# Disaster Response Pipeline
Python code for a disaster response pipeline.

## Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Directory Structure](#directoryStructure)
4. [Design](#design)
5. [Results](#results)

## Installation <a name="installation"></a>

This project was written in Python 3.6, using a Jupyter Notebook on Anaconda. The relevant Python packages for this project are as follows:

- numpy
- pandas
- matplotlib


## Project Motivation <a name="motivation"></a>


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
    - models /  
        - train_classifier.py  
    - uml /
        - Use cases.png  

## Design <a name="design"></a>

### Main Uses Cases
There are 3 main use cases:

- Import data
- Train model
- Classify message

<p align="center">
    <img src="./uml/Use cases.png" width="800" title="Main use cases." alt="Main use cases.">
</p>

## Results <a name="results"></a>
