# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:51:13 2021
@author: Jaju Peter
"""
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Load dataset
# -----------------------------
try:
    house_data = pd.read_csv('kc_house_data.csv', encoding='ISO-8859-1')
except FileNotFoundError:
    st.error("❌ kc_house_data.csv file not found. Please ensure the file is in the correct directory.")
    st.stop()

# -----------------------------
# Modern Title
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>
        🏡 House Price Prediction App
    </h1>
    <h4 style='text-align: center; color: gray;'>
        Powered by Deep Learning • King County, Washington
    </h4>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# App Description
# -----------------------------
st.markdown(
    """
    <div style="
        background-color:#F8F9F9;
        padding:15px;
        border-radius:10px;
        border: 1px solid #D6DBDF;
        ">
        <h3 style="color:#2E4053;">📘 About this App</h3>
        <p style="color:#5D6D7E;">
        Buying or selling a home is one of the biggest financial decisions people make — 
        and pricing it right matters.<br><br>
        This app uses <b>real-world housing data from King County, Washington (2014–2015)</b> 
        and a <b>deep learning regression model</b> to estimate home values.
        </p>
        <ul style="color:#5D6D7E;">
            <li>⚙️ Interactive sidebar to set <b>18 property attributes</b> (bedrooms, bathrooms, square footage, year built, location, and more)</li>
            <li>⚡ Instant predictions powered by AI</li>
            <li>🔍 Dataset summary & debug info available for deeper insights</li>
        </ul>
        <p style="color:#5D6D7E;">
        🎯 Whether you’re a homeowner, real estate investor, or data enthusiast, 
        this app shows how <b>AI meets real estate valuation</b>.
        </p>
    </div>
    """, unsafe_allow_html=True
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.markdown(
    "<h2 style='color:#2E86C1;'>⚙️ Input Parameters</h2>", 
    unsafe_allow_html=True
)

if st.checkbox('📊 Show Summary of Dataset'):
    st.write(house_data.describe())

# -----------------------------
# Load model + recreate scalers
# -----------------------------
@st.cache_resource
def load_models_and_scalers():
    try:
        try:
            model_ann = tf.keras.models.load_model("ann_model.h5")
        except:
            model_ann = tf.keras.models.load_model("ann_model.hdf5")

        Xa = house_data.drop(['price', 'date', 'id', 'view'], axis=1)
        Y = house_data['price'].values.reshape(-1,1)

        scalerX = MinMaxScaler()
        scalerY = MinMaxScaler()

        scalerX.fit(Xa)
        scalerY.fit(Y)

        return model_ann, scalerX, scalerY, True
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, False

model_ann, scalerX, scalerY, models_loaded = load_models_and_scalers()

# -----------------------------
# Sidebar Input Function
# -----------------------------
def user_input_parameters():         
    bedrooms = st.sidebar.slider("1. No of bedrooms?", 0, 12, 4)      
    bathrooms = st.sidebar.slider("2. No of bathrooms?", 0, 15, 5)    
    sqft_living = st.sidebar.slider("3. Square footage of the house?", 500, 15000, 2000) 
    sqft_lot = st.sidebar.slider("4. Square footage of the lot?", 500, 170000, 1200)
    floors = st.sidebar.slider("5. No of floors?", 0, 5, 3)
    condition = st.sidebar.slider("6. Overall condition? (1 = poor, 5 = excellent)", 0, 5, 3)
    grade = st.sidebar.slider("7. Grade (1 = poor, 13 = excellent)", 0, 13, 6)
    sqft_above = st.sidebar.slider("8. Sqft above basement?", 200, 12000, 5000)
    sqft_basement = st.sidebar.slider("9. Sqft of basement?", 0, 7000, 2500)
    yr_built = st.sidebar.slider("10. Year Built?", 1900, 2019, 2009)
    
    yr_renovated = st.sidebar.radio('11. Renovation?', ('Known', 'Unknown'))
    if yr_renovated == 'Unknown':
        yr_renovated = 0
    else:
        yr_renovated = st.sidebar.slider("Year Renovated?", 1900, 2019, 2010)

    zipcode = st.sidebar.slider("12. Zipcode?", 98001, 98288, 98250)
    lat = st.sidebar.slider("13. Latitude?", 47.000100, 47.800600, 47.560053, 0.000001, "%g")
    long = st.sidebar.slider("14. Longitude?", -122.6000000, -121.300500, -122.213896, 0.000001, "%g")
    sqft_living15 = st.sidebar.slider("15. Sqft living (2015)?", 200, 12000, 3500)
    sqft_lot15 = st.sidebar.slider("16. Sqft lot (2015)?", 200, 12000, 3700)
    
    waterfront = st.sidebar.radio('17. Waterfront View?', ('Yes', 'No'))
    waterfront = 1 if waterfront == 'Yes' else 0
    
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
    
    return pd.DataFrame(features, index=[0])

# -----------------------------
# Get User Input
# -----------------------------
df = user_input_parameters()

st.subheader('📥 Selected Input Parameters')
st.write(df)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_ann(input_df):
    try:
        df_array = np.array(input_df)
        X_scaled = scalerX.transform(df_array)   # (1, 17)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # (1, 1, 17)

        prediction = model_ann.predict(X_scaled, verbose=0)

        if prediction.ndim > 2:
            prediction = prediction.reshape(-1, 1)

        prediction_original = scalerY.inverse_transform(prediction)
        return float(prediction_original[0][0])
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return 0.0

# -----------------------------
# Prediction Section
# -----------------------------
if models_loaded:
    if st.button('🔮 PREDICT PRICE'):
        with st.spinner('⏳ Running prediction...'):
            house_price = predict_ann(df)
            
        if house_price > 0:
            st.markdown(
                f"""
                <div style="
                    background-color:#E8F8F5;
                    padding:20px;
                    border-radius:10px;
                    border: 2px solid #1ABC9C;
                    text-align:center;
                    ">
                    <h2 style="color:#1ABC9C;">
                    💰 Predicted Price: ${house_price:,.2f}
                    </h2>
                    <p style="color:#117A65;">
                    (Based on Deep Learning Algorithm)
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.error("❌ Prediction failed. Please check the model files and try again.")
else:
    st.error("""
    ❌ Could not load the required model files. Please ensure the following files exist in your directory:
    - ann_model.h5 (or ann_model.hdf5)
    - kc_house_data.csv (needed to recreate scalers)
    """)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: gray;'>
        🔗 <a href="https://github.com/jajupeter/House-Price-prediction-app" target="_blank">View Source Code on GitHub</a>
    </p>
    """, unsafe_allow_html=True
)