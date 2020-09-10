import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from PIL import Image

from titanic_req.titanic_model import titanic

def titanic_data(sidebar_slots, predict_button):
    model_titanic = titanic()
    dtree, svc, name = model_titanic.input(sidebar_slots, predict_button)

    if predict_button:
        message = f"{name} unfortunately will **_not_ Survive**" if dtree[0] == 0 else f"{name} **_will_ Survive**"
        st.subheader("DECISION TREE PREDICTION")
        st.markdown(message)
        message = f"{name} unfortunately will **_not_ Survive**" if svc[0] == 0 else f"{name} **_will_ Survive**"
        st.subheader("SUPPORT VECTOR MACHINE PREDICTION")
        st.markdown(message)

def cars_prediction(mainwindow_slots, sidebar_slots, predict_button):
    buying = sidebar_slots[0].selectbox("Buying Value", options = ["vhigh", "low", "high", "med"])
    maint = sidebar_slots[1].selectbox("Maintanance", options = ["vhigh", "low", "high", "med"])
    doors = sidebar_slots[2].number_input("Number of Doors", min_value = 2, max_value = 5, value = 4, step = 1)
    persons = sidebar_slots[3].selectbox("Number of Persons", options = [2, 4, 5])
    lug_boot = sidebar_slots[4].selectbox("Lugguage Boot Space", options = ["big", "small", "med"])
    safety = sidebar_slots[5].selectbox("Safety", options = ["low", "high", "med"])

    dat = {"buying" : buying,
            "maint" : maint,
            "doors" : doors,
            "persons" : persons,
            "lug_boot" : lug_boot,
            "safety" : safety}

    dat = pd.DataFrame(dat, index = [0])

    buying = 3 if buying == "vhigh" else buying
    buying = 2 if buying == "low" else buying
    buying = 1 if buying == "high" else buying
    buying = 0 if buying == "med" else buying

    maint = 3 if maint == "vhigh" else maint
    maint = 2 if maint == "low" else maint
    maint = 1 if maint == "high" else maint
    maint = 0 if maint == "med" else maint

    doors = 1 if doors == 4 else doors
    doors = 0 if doors == 5 else doors

    persons = 1 if persons == 4 else persons
    persons = 0 if persons == 5 else persons

    lug_boot = 2 if lug_boot == "big" else lug_boot
    lug_boot = 1 if lug_boot == "small" else lug_boot
    lug_boot = 0 if lug_boot == "med" else lug_boot

    safety = 2 if safety == "low" else safety
    safety = 1 if safety == "high" else safety
    safety = 0 if safety == "med" else safety

    st.write("""
    # Cars Evaluation
    """)
    image_path = os.path.join(os.getcwd(), "support_models")
    image_path = os.path.join(image_path, "images")
    image_path = os.path.join(image_path, "cars.jpg")
    banner = Image.open(image_path)
    st.image(banner, use_column_width = True)

    st.write("Enter the details of your car to view the predictions")

    if predict_button:
        st.subheader("User Input features")
        st.write(dat)

        arr = np.array([buying, maint, doors, persons, lug_boot, safety])

        file_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(file_path, "dtree_cars.sav")

        dtree_cars = joblib.load(file_path)

        if predict_button:
            st.write("""
            ## Decision Tree Prediction
            """)
            prediction_cars = dtree_cars.predict(arr.reshape(1,-1))
            message = "### This Car is "
            message = message + "***Un Acceptable***" if prediction_cars == "unacc" else message
            message = message + "***Un Acceptable***" if prediction_cars == "acc" else message
            message = message + "***Un Acceptable***" if prediction_cars == "good" else message
            message = message + "***Un Acceptable***" if prediction_cars == "v-good" else message

            st.write(message)