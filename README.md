# DisasterM

**A Data based approach for sorting important messages**

**Installation and Library Used**

Following Libraries are used for this project:
1. Python
2. SQL
3. PANDAS
4. NumPy
5. Plotly
6. Pickle
7. Scikit Learn
8. Regex
9. BootStrap

These libraries can be installed using simple pip command or conda (if you are using Jupyter). 


**Project Motivation:**

This project is a part of my course curriculum in UDACITY DataScientist NanoDegree. 

**File Description**

This project contains three folders as under:
1.**data** : This folder contains two sets of data viz. categories and messages. **Categories** dataset describes the category in which the messages are divided.         **Messages** contains the datset with actual messages. Both the datset have a common Id on which we have merged to form a single dataframe. 
         This folder also contains a python file which can be opened with the help of Terminal or a Jupyter Notebook. It covers our data cleaning part. After cleaning            the file we have saved it as a DataBase which is our fourth file in the folder named DisasterResponse.db
2.**models** : Here is our training model. We have employed Forrest Random Classifier with Grid Search for cross validation.
3.**app** : This folder contains our app which can be used as a front-end to classify test messages.

**How to Interact with the project:**

Most of the code for cleaning can be done on Jupyter/Colab. For interacting with the app please follow as below: 

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    - To run app
        `python run.py`
    - Go to http://0.0.0.0:3001/
        
**Technical Details:**

Random Forrest ensemble method is used for our model for the classification. Grid Search technique employed with cv = 3 in our model.

**Summary:**

With this model, we primary looking for three basic questions:

1.What kind of messages do we have in the datset and how are they distributed as per the genres?

Inference: Majority of the messages came from the News and Direct Message genre. As these are the primary source of information received in case of any adversity. This can be seen from the Bar chart below:

![Genre](https://github.com/Prinkesh39/DisasterM/blob/main/Message_genre.png?raw=true)

2. What are the categories of the messages we have divided our dataset into:

Inference: As per below we have divided the messages into 36 categories as per the scatter plot below:

![Categories_Scatter](https://github.com/Prinkesh39/DisasterM/blob/main/Message_Categories_Scatter.png?raw=true)

![Categories_Pie](https://github.com/Prinkesh39/DisasterM/blob/main/Message_cat_pie.png?raw=true)


3. What is the overall precision we have achieved in correctly classifying a message as per the above categories.
Inference: As can be seen from our model we have reached average precision value of 73% using GridSearchCV method


**Licensing, Authors, Acknowledgements, etc.***

Thanks to Udacity for starter code for the web app.
