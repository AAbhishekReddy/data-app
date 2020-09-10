import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from PIL import Image

def beer_data(sidebar_slots, predict_button):
    beer_name = sidebar_slots[0].text_input("Beer Name:", value = "British Empire")
    review_aroma = sidebar_slots[1].number_input("Aroma Review", min_value = 1, max_value = 5)
    review_appearence = sidebar_slots[2].number_input("Aroma Appereance", min_value = 1, max_value = 5)
    review_palate = sidebar_slots[3].number_input("Palate Appereance", min_value = 1, max_value = 5)
    review_taste = sidebar_slots[4].number_input("Taste Appereance", min_value = 1, max_value = 5)
    beer_abv = sidebar_slots[5].number_input("Beer ABV", min_value = 1, max_value = 100)

    beers = {"beer_name": beer_name,
             "review_aroma": review_aroma,
             "review_palate": review_palate,
             "review_taste": review_taste,
             "review_appereance": review_appearence,
             "beer_abv": beer_abv}
    
    beers = pd.DataFrame(beers, index = [0])

    st.write("""
        # Grab a beer....
        """)
    image_path = os.path.join(os.getcwd(), "support_models")
    image_path = os.path.join(image_path, "images")
    image_path = os.path.join(image_path, "craft-beer.jpg")
    banner = Image.open(image_path)
    st.image(banner, use_column_width = True)

    st.write("Enter the beer details in the sidebar to view the evaluation of the beer")

    if predict_button:
        st.subheader("User Input features")
        st.dataframe(beers)

        arr = np.array([review_aroma, review_palate, review_taste])

        file_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(file_path, "linear_regression.sav")

        beer_lr = joblib.load(file_path)

        prediction_beers = beer_lr.predict(arr.reshape(1,-1))
        st.write("### Linear Regression Prediction")
        message = f"The overall rating for {beer_name} is: {prediction_beers[0]}" 
        st.write(message)


def nyse_data(sidebar_slots, predict_button):
    symbol = sidebar_slots[0].selectbox("Company Symbol", options = ["AMZN", "AAPL", "MSFT"])
    open_num = sidebar_slots[1].number_input("Opening Value")
    high = sidebar_slots[2].number_input("Highest Value")
    low = sidebar_slots[3].number_input("Lowest Value")

    data = {"symbol": symbol,
            "open": open_num,
            "high": high,
            "low": low}
    data = pd.DataFrame(data, index = [0])

    st.write("""
        # NYSE Stock Prediction
        """)
    image_path = os.path.join(os.getcwd(), "support_models")
    image_path = os.path.join(image_path, "images")
    image_path = os.path.join(image_path, "nyse.jpg")
    banner = Image.open(image_path)
    st.image(banner, use_column_width = True)

    st.write("Enter the stock details in the sidebar to view the predictions")

    if predict_button:
        st.subheader("User Input features")
        st.dataframe(data)

        file_path = os.path.dirname(os.path.realpath(__file__))

        if symbol == "AMZN":
            file_path = os.path.join(file_path, "amazon_regression.sav")
        elif symbol == "MSFT":
            file_path = os.path.join(file_path, "microsoft_regression.sav")
        else:
            file_path = os.path.join(file_path, "apple_regression.sav")
        
        reg = joblib.load(file_path)

        arr = np.array([open_num, low, high])

        prediction = reg.predict(arr.reshape(1,-1))

        st.write("### Linear Regression Prediction")
        message = f"The closing value of {symbol} could be: {prediction[0]}" 
        st.write(message)