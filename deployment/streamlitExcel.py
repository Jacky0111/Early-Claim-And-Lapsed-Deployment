
#importing required libraries
# streamlit run your_script.py --server.maxUploadSize=1028
import streamlit as st
import pandas as pd
from io import StringIO 
import joblib

# Load saved model
from Medical_Claim import preprocessing

model = joblib.load(open('//TMLPC18108/CP_Shared/lgbm_model_auc.pkl','rb'))

#adding a file uploader to accept multiple CSV files

uploaded_files = st.file_uploader("Please choose a CSV file", accept_multiple_files=True)

preprocessing(uploaded_files)

for file in uploaded_files:

    bytes_data = file.read()

    st.write("File uploaded:", file.name)

    st.code(float(model.predict([[file]])))
    # st.write(bytes_data)