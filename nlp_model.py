# Natural Language Processing (NLP) Model
#A python file that predicts loan purspose based on loan description given by the borrower

#Imports

#Data
import pandas as pd
import numpy as np


#Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Text
import string
from nltk.corpus import stopwords

#Other
import pickle

#General
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#NLP - Related
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#Getting the csv file's name with the wrangled data
df=pd.read_csv('data/xyz_corp_lending_data.csv',sep='\t')

#Including only the non-na data for desc and getting only the
#  desc and purpose columns
df=df[df['desc'].isna()==False][['desc','purpose']]

#A function that takes in a string of text, then remove all punctuation,
#  removes all stopwords, returns a list of the cleaned text
def text_process(mess):

    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords (very common words)
    return [word for word in nopunc.split() if word.
    lower() not in stopwords.words('english')]

    #NLP model predicting loan purpose based on loan description
    #  given by the borrower.
#The model's input is the method used for NLP. If set to initial
#  it just uses count_vectorizer
#if set to anything else it also uses a TfidfTransformer. It is
#  observed that the model runs better
#with the 'initial' input.

def get_model_nlp(method='initial'):


    #Using the text processing function
    #df['desc']=df['desc'].apply(text_process)

    #Creating X and y variables for input and output
    X=df['desc']
    y=df['purpose']

    if method=='initial':
        #Vectorizing the data
        cv = CountVectorizer(analyzer=text_process)
        X = cv.fit_transform(X)

        #Splitting the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X,
         y,test_size=0.3)

        #Using the Naive-Base Binomial algorith for prediction
        model_nlp = MultinomialNB()
        model_nlp.fit(X_train,y_train)
        predictions = model_nlp.predict(X_test)
    else:
        #Using a Pipeline including a TfidfTransformer
        model_nlp = Pipeline([
        ('bow', CountVectorizer(analyzer=text_process)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB()),]) 

        #Splitting the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X,
         y,test_size=0.3)

        #Using the Naive-Base Binomial algorith for prediction
        model_nlp.fit(X_train,y_train)
        predictions = model_nlp.predict(X_test) 
    
    #Printing useful metrics
    #print(confusion_matrix(y_test,predictions))
    print('NLP Model')
    print(classification_report(y_test,predictions))
    return model_nlp

#Running the function 
model_nlp=get_model_nlp(method='initial')
#pickle.dump(model_nlp,open('model_nlp','wb'))