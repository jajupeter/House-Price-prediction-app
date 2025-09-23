# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:51:13 2021

@author: Jaju Peter
"""
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load dataset
try:
    house_data = pd.read_csv('kc_house_data.csv', encoding='ISO-8859-1')
except FileNotFoundError:
    st.error("kc_house_data.csv file not found. Please ensure the file is in the correct directory.")
    st.stop()

# App title
st.write("""
    # House Price Prediction App
    *Washington DC, USA*
    """)
    
html_temp = """
    <div style="background-color: #f0f0f5; padding: 5px">
    <h3 style="color:#666666;text-align:left; line-height: 1.5">
    <p> <b> bout this App
Buying or selling a home is one of the biggest financial decisions people make — and pricing it right matters.

This app uses real-world housing data from King County, Washington (2014–2015) and a deep learning regression model to estimate home values.

🔑 Key Features:
	•	Interactive sidebar to set 18 property attributes (bedrooms, bathrooms, square footage, year built, location, and more).
	•	Instant predictions powered by AI.
	•	Transparent insights — view dataset summary and debug information if you want to peek under the hood.

🎯 Whether you’re a homeowner, real estate investor, or data enthusiast, this app gives you a taste of how AI meets real estate valuation.</p></h3>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

st.sidebar.header('User Input Parameters') 
    
if st.checkbox('Show Summary of Dataset'):
    st.write(house_data.describe())

# Load models and create scalers from training data
@st.cache_resource
def load_models_and_scalers():
    try:
        # Load the model
        try:
            model_ann = tf.keras.models.load_model("ann_model.h5")
        except:
            model_ann = tf.keras.models.load_model("ann_model.hdf5")
        
        # Recreate scalers from the original data (same as in your notebook)
        # This matches exactly what you did in your training
        Xa = house_data.drop(['price', 'date', 'id', 'view'], axis=1)
        Y = house_data['price'].values.reshape(-1,1)
        
        # Create and fit scalers
        scalerX = MinMaxScaler()
        scalerY = MinMaxScaler()
        
        scalerX.fit(Xa)
        scalerY.fit(Y)
            
        return model_ann, scalerX, scalerY, True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, False

model_ann, scalerX, scalerY, models_loaded = load_models_and_scalers()

def user_input_parameters():         
    bedrooms = st.sidebar.slider("1. No of bedrooms?", 0, 12, 4)      
    bathrooms = st.sidebar.slider("2. No of bathrooms?", 0, 15, 5)    
    sqft_living = st.sidebar.slider("3. Square footage of the house?", 500, 15000, 2000) 
    sqft_lot = st.sidebar.slider("4. Square footage of the lot?", 500, 170000, 1200)
    floors = st.sidebar.slider("5. No of floors?", 0, 5, 3)
    condition = st.sidebar.slider("7. Overall condition? (1 indicates worn out property and 5 excellent)", 0, 5, 3)
    grade = st.sidebar.slider("8. Overall grade based on King County grading system? (1 poor ,13 excellent)", 0, 13, 6)
    sqft_above = st.sidebar.slider("9. Square footage above basement?", 200, 12000, 5000)
    sqft_basement = st.sidebar.slider("10. Square footage of the basement?", 0, 7000, 2500)
    yr_built = st.sidebar.slider("11. Year Built?", 1900, 2019, 2009)
    
    yr_renovated = st.sidebar.radio('12. Year renovated?', ('Known', 'Unknown'))
    if yr_renovated == 'Unknown':
        yr_renovated = 0
    else:
        yr_renovated = st.sidebar.slider("Year Renovated?", 1900, 2019, 2010)

    zipcode = st.sidebar.slider("13. Zipcode of the house?", 98001, 98288, 98250)
    lat = st.sidebar.slider("14. Location of House (latitude)?", 47.000100, 47.800600, 47.560053, 0.000001, "%g")
    long = st.sidebar.slider("15. Location of House (longitude)?", -122.6000000, -121.300500, -122.213896, 0.000001, "%g")
    sqft_living15 = st.sidebar.slider("16. Square footage of the house in 2015?", 200, 12000, 3500)
    sqft_lot15 = st.sidebar.slider("17. Square footage of the lot in 2015?", 200, 12000, 3700)
    
    waterfront = st.sidebar.radio('18. House has Waterfront View?', ('Yes', 'No'))
    if waterfront == 'Yes':
        waterfront = 1
    else:
        waterfront = 0 
    
    features = {
        'bedrooms': bedrooms, 
        'bathrooms': bathrooms, 
        'sqft_living': sqft_living, 
        'sqft_lot': sqft_lot,
        'floors': floors, 
        'waterfront': waterfront,
        'condition': condition, 
        'grade': grade, 
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement, 
        'yr_built': yr_built, 
        'yr_renovated': yr_renovated, 
        'zipcode': zipcode, 
        'lat': lat, 
        'long': long, 
        'sqft_living15': sqft_living15,
        'sqft_lot15': sqft_lot15
    }
    
    feat = pd.DataFrame(features, index=[0])
    return feat

# Get user input
df = user_input_parameters()

st.subheader('User Input parameters')
st.write(df)

# Debug information (optional)
if st.checkbox('Show Debug Info') and models_loaded:
    try:
        st.write("Model input shape:", model_ann.input_shape)
        st.write("Input data shape:", df.shape)
        st.write("Scaler feature count:", scalerX.n_features_in_)
    except Exception as e:
        st.write(f"Debug info error: {e}")

# Prediction function
def predict_ann(input_df):
    try:
        # Convert dataframe to numpy array
        df_array = np.array(input_df)

        # ✅ Scale first (2D input)
        X_scaled = scalerX.transform(df_array)   # shape (1, 17)

        # ✅ Then reshape for the model (if model expects 3D)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # (1, 1, 17)

        # Make prediction
        prediction = model_ann.predict(X_scaled, verbose=0)

        # Ensure shape matches scalerY
        if prediction.ndim > 2:
            prediction = prediction.reshape(-1, 1)

        # ✅ Inverse scale back to original price range
        prediction_original = scalerY.inverse_transform(prediction)

        return float(prediction_original[0][0])
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0.0



# Prediction section
st.text("")

if models_loaded:
    if st.button('PREDICT PRICE'):
        with st.spinner('Making prediction...'):
            house_price = predict_ann(df)
            
        if house_price > 0:
            st.success(f"**${house_price:,.2f}** - *based on Deep Learning Algorithm*")
        else:
            st.error("Prediction failed. Please check the model files and try again.")
else:
    st.error("""
    Could not load the required model files. Please ensure the following files exist in your directory:
    - ann_model.h5 (or ann_model.hdf5)
    - kc_house_data.csv (needed to recreate scalers)
    """)

# Source code link
url = '[SOURCE CODE](https://github.com/jajupeter/House-Price-prediction-app)'
st.markdown(url, unsafe_allow_html=True)