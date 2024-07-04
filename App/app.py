import re
import pandas as pd
import numpy as np
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import csv
import warnings
import streamlit as st

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load data
training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

# Train models
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)
print(scores.mean())

model = SVC()
model.fit(x_train, y_train)
print("for svm: ")
print(model.score(x_test, y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()
symptoms_dict = {symptom: index for index, symptom in enumerate(x)}

# Functions to get severity, description, and precaution
def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]

def getSeverityDict():
    global severityDictionary
    with open('symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) < 2:
                continue  # Skip rows that do not have at least two columns
            try:
                severityDictionary[row[0]] = int(row[1])
            except ValueError:
                continue 

def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

getSeverityDict()
getDescription()
getprecautionDict()

# Function to predict the secondary condition
def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)
    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[symptoms_dict[item]] = 1
    return rf_clf.predict([input_vector])

# Streamlit interface
# Streamlit interface
st.title("HealthCare ChatBot")
st.write("Enter your symptoms below:")

# Get user's name
user_name = st.text_input("Enter your name")

symptom1 = st.text_input("Symptom 1")
symptom2 = st.text_input("Symptom 2")
symptom3 = st.text_input("Symptom 3")
symptom4 = st.text_input("Symptom 4")
symptom5 = st.text_input("Symptom 5")

if st.button("Predict Disease"):
    symptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]
    input_vector = np.zeros(len(cols))
    for sym in symptoms:
        if sym in cols:
            input_vector[cols.get_loc(sym)] = 1

    prediction = clf.predict([input_vector])
    disease = le.inverse_transform(prediction)[0]
    
    st.write(f"Hello, {user_name}!")
    st.write(f"The predicted disease is: {disease}")
    
    # Show description and precautions
    if disease in description_list:
        st.write(f"Description: {description_list[disease]}")
        precautions = precautionDictionary.get(disease, [])
        if precautions:
            st.write("Take the following precautions:")
            for i, precaution in enumerate(precautions):
                st.write(f"{i+1}. {precaution}")
    
    st.write("Please remember, this prediction is based on the symptoms entered.")
    st.write("It is important to consult a doctor for a proper diagnosis and treatment.")


# To run the app, use the command: streamlit run app.py
