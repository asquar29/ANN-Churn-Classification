import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np


# load the pickle file
model = load_model('model.h5')

# load encoder and scaler
with open('OHE_geo.pkl','rb') as file:
    label_encoder_geo= pickle.load(file)

with open('label_encoder_gender.pkl','rb')as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)



# Steamlit app
st.title('Customer Churn Prediction')

# user i/p
geography = st.selectbox('Geography',label_encoder_geo.categories[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.slider('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_number = st.selectbox('Is Active Member',[0,1])


# prepare  i/p data
input_data =pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' :[num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember':[is_active_number],
    'EstimatedSalary':[estimated_salary]
})

# Onehotencode '    ' 
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=label_encoder_geo.get_feature_names_out(['Geography']))

#concatination with OHE data
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)


# Scaling the i/p data
input_scale = scaler.transform(input_data)

# Prediction 
y_pred = model.predict(input_scale)
prediction_prb = y_pred[0][0]

st.write(f'Churn Probability: {prediction_prb:.2f}')
if (prediction_prb>0.5):
    st.write("Customer is likely to churn")

else:
    st.write('The customer is not likely to churn')

