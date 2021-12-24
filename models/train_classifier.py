import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import sqlalchemy as sa
import pickle
#Loading data from the Database using SQAlchemy Engine
def load_data(database_filepath):
    """
    Loads data from SQLite database.
    
    Parameters:
    database_filepath: Filepath to the database
    
    Returns:
    X: Features
    Y: Target
    Category-names
    """    
    engine = sa.create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_messages', engine)
    X = df['message'] #Feature for our dataset
    Y = df[df.columns].drop(['id', 'message', 'original', 'genre'], axis = 1) #Target Variable
    Y.related.replace(2,1,inplace=True) #related category had three values 0, 1, 2
    category_names = list(df.columns[4:]) 
    return X, Y, category_names
#Cleaning the data
def tokenize(text):
    """
    Tokenizes and lemmatizes text.
    
    Parameters:
    text: Text to be tokenized
    
    Returns:
    clean_tokens: Returns cleaned tokens 
    """    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    #tokenize text
    tokens = word_tokenize(text)
    #Initailize Lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    #iterate through each token
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() #lemmatize, normalise, strip leading/trailing white space
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Builds classifier and tunes model using GridSearchCV.
    
    Returns:
    cv: Classifier 
    """        
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)), #This 
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])
    
    parameters = {
              'clf__estimator__n_estimators': [10, 20, 50],
              'clf__estimator__min_samples_split' : [2, 5, 10],
              }

    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs = -1, cv = 3, verbose = 2)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the performance of model and returns classification report. 
    
    Parameters:
    model: classifier
    X_test: test dataset
    Y_test: labels for test data in X_test
    
    Returns:
    Classification report for each column
    """    
    y_pred = model.predict(X_test)
    
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))    
    


def save_model(model, model_filepath):
    """ Exports the final model as a pickle file."""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Builds the model, trains the model, evaluates the model, saves the model."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
