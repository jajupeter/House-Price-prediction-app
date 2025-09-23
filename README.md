# 🏡 House Price Prediction App  
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen?logo=streamlit)](https://real-estate-house-price-prediction-app.streamlit.app/)  
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)  
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#)  

---

## 📌 Project Overview
The **House Price Prediction App** is an AI-powered interactive tool built with **Streamlit** that estimates residential property prices in **King County, Washington (2014–2015 dataset)**.  
Using a trained **Deep Learning regression model**, the app demonstrates how machine learning can be applied to real-estate valuation in a user-friendly web interface.  

🔗 **Live Demo**: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://real-estate-house-price-prediction-app.streamlit.app/) 

---

## ❓ Problem Statement
Accurately pricing a home is one of the most critical steps in real estate.  
Traditional valuation methods rely heavily on manual appraisals or simplistic statistical models, which:  
- Are often **time-consuming**  
- Can be **subjective**  
- Struggle to capture **non-linear relationships** in housing features  

This project addresses the challenge by applying **AI models** to predict house prices based on **18 property features** such as square footage, number of bedrooms/bathrooms, year built, renovation status, and geolocation.  

---

## 🔬 Methodology – Approach
1. **Data Collection**:  
   - Dataset: *King County Housing Data (2014–2015)*  
   - Preprocessing: removed non-essential fields (`id`, `date`, `view`)  

2. **Feature Engineering**:  
   - Selected **18 key predictors** (bedrooms, bathrooms, sqft, grade, location, etc.)  
   - Normalized input data using **MinMaxScaler**  

3. **Modeling**:  
   - Built and trained an **Artificial Neural Network (ANN)** using **TensorFlow/Keras**  
   - Optimized for regression on price values  

4. **Deployment**:  
   - Packaged the model with **Streamlit**  
   - Deployed on **Streamlit Cloud** for instant access  

---

## ⚙️ Tech Stack
- **Programming Language**: Python 3.10+  
- **Frameworks/Libraries**:  
  - [Streamlit](https://streamlit.io/) – Interactive web UI  
  - [TensorFlow/Keras](https://www.tensorflow.org/) – Deep learning model  
  - [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) – Data manipulation  
  - [scikit-learn](https://scikit-learn.org/) – Preprocessing (MinMaxScaler)  
- **Deployment**: Streamlit Cloud  
- **Version Control**: Git + GitHub  

---

## ✨ App Features
- 🎛 **Interactive Sidebar**: Adjust 18 property attributes with sliders & radio buttons  
- 📊 **Dataset Insights**: View summary statistics of the training dataset  
- 🤖 **AI Predictions**: Instantly estimate house prices using the trained deep learning model  
- 🎨 **Modern UI/UX**: Clean visuals with styled components and responsive design  
- 🔍 **Debug Mode**: Optional visibility into input shapes, scalers, and model info  
- 💻 **Open Source**: Full code available for learning and extension  

---






