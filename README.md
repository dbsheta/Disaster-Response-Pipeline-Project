# Disaster Response Pipeline Project

## Introduction:
In the Project, I have a data set containing real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

The project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## Installation:
Python 3.x is required to run this Project. It uses the following libraries which can be installed using pip/conda:
1. Pandas
2. Numpy
3. Matplotlib
4. Plotly
5. Sklearn
6. NLTK

## File Structure:
Here's the file structure of the project:
- app
    - template
    - master.html (main page of web app)
    - go.html  (classification result page of web app)
    - run.py  (Flask file that runs app)

- data
    - disaster_categories.csv  (data to process)
    - disaster_messages.csv  (data to process)
    - process_data.py
    - InsertDatabaseName.db   (database to save clean data to)

- models
    - train_classifier.py
    - classifier.pkl  (saved model) 

- README.md


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. You can also download pre-trained model from [here](https://yadi.sk/d/iFtlqhotcM9Kpg).
2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

