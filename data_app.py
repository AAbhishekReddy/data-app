import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from titanic_req.titanic_model import titanic
from support_models import classsification, regression

# Global variables
model_select_preview = "Classification Models"

# Creating the support functions
def classification_model(mainwindow_slots, sidebar_slots):
    

    data_select = st.radio("Select the data that you want to apply the model on", options = ["Titanic Dataset", "Car Evaluation Dataset"])
    if data_select == "Titanic Dataset":
        classsification.titanic_data(sidebar_slots, predict_button)

    else:
        classsification.cars_prediction(mainwindow_slots, sidebar_slots, predict_button)


def regression_model(mainwindow_slots, sidebar_slots):
    data_select = st.radio("Select the data set to which you want to apply the model.", options = ["Beer Review Dataset", "New York Stock Exchange Dataset"])

    if data_select == "Beer Review Dataset":
        regression.beer_data(sidebar_slots, predict_button)
    else:
        regression.nyse_data(sidebar_slots, predict_button)


if __name__ == "__main__":
    # Creating the side bar empty fields.
    sidebar_slots = []

    for i in range(10):
        sidebar_slots.append(st.sidebar.empty())

    predict_button = sidebar_slots[8].button("Predict") 

    # Creating the main page Elements
    st.write("""
    # This is an Interactive app to play around with various Data Models
    """)

    # Main window slots:
    # 1. Select the type of Models ("Classification or Regeression")
    # 2. Select the data set in the options.
    # 3. Display the objective of the Models
    # 4. Display the use Input data.
    # 5. Display the option for the models in the same data.
    # 6. Display the output or the prediction.

    # Creating a radio button for slecting the type of models.
    st.write("## Select the type of models that you want to play with below")
    model_select = st.radio("Select the models of your choice", options = ["Classification Models", "Regression Models"])

    # Creating the mainwindow slots.
    mainwindow_slots = []

    for i in range(10):
        mainwindow_slots.append(st.empty())

    if model_select == "Classification Models":
        classification_model(mainwindow_slots, sidebar_slots)

        
    else:
        regression_model(mainwindow_slots, sidebar_slots)

