# Streamlit Script

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained models and encoders
with open('product_name_prediction_model.pkl', 'rb') as f:
    model_product_name = pickle.load(f)

with open('price_prediction_model.pkl', 'rb') as f:
    model_price = pickle.load(f)

with open('le_product_name.pkl', 'rb') as f:
    le_product_name = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the numerical columns
numerical_columns = ['Weight']

# Streamlit UI
st.title('Furniture Recommendation Sysytem')

material = st.text_input('Enter the material:')
color = st.text_input('Enter the color:')
category = st.text_input('Enter the category:')
weight = st.number_input('Enter the weight:')

if st.button('Predict Product Name and Price'):
    if material and color and category and weight:
        # Encode categorical features
        try:
            material_enc = label_encoders['Material'].transform([material])[0]
            color_enc = label_encoders['Color'].transform([color])[0]
            category_enc = label_encoders['Category'].transform([category])[0]
        except KeyError as e:
            st.write(f'Error: {e} not found in the label encoders.')
            st.stop()
        except ValueError as e:
            st.write(f'Error: {e}')
            st.stop()

        # Scale numerical features
        input_data = pd.DataFrame({
            'Material': [material_enc],
            'Color': [color_enc],
            'Category': [category_enc],
            'Weight': [weight]
        })
        input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])
        
        # Predict ProductName
        product_name_encoded = model_product_name.predict(input_data)[0]
        product_name = le_product_name.inverse_transform([product_name_encoded])[0]
        
        # Predict Price
        price = model_price.predict(input_data)[0]
        
        st.write(f'The predicted product name is: {product_name}')
        st.write(f'The predicted price is: ${price:.2f}')
    else:
        st.write('Please enter all the required details.')

# Save the Streamlit script as app.py and run using:
# streamlit run app.py
