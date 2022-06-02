import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from streamlit_lottie import st_lottie
from sklearn.ensemble import RandomForestClassifier

st.write('# Early Claim and Lapsed Prediction')

st.sidebar.header('User Input Features')


# Collects user input features into dataframe
def userInputFeatures():
    risk_code = st.sidebar.selectbox('Risk Code',
                                     ('RTH1', 'RTI1', 'RTM1', 'RTN1', 'RTP1', 'RTP1', 'RTX1', 'UNHA', 'UNHC',
                                      'UNHD', 'UNHE', 'UNHG', 'UNHH', 'UNHJ', 'UNHK', 'UNHL', 'UNHP'))

    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))

    occupation_class = st.sidebar.selectbox('Occupation Class', ('A', 'B', 'C', 'D'))

    race = st.sidebar.selectbox('Race', ('BN', 'CH', 'IN', 'MA', 'OT'))

    marital_status = st.sidebar.selectbox('Marital Status',
                                          ('Single', 'Married', 'Widowed', 'Divorced', 'Not Disclosed', 'Unknown'))

    entry_age = st.sidebar.slider('Entry Age', 0, 100)

    smoker = st.sidebar.selectbox('Smoker', ('Smoker', 'Non-Smoker'))

    risk_sum_assured = st.sidebar.selectbox('Risk Sum Assured', ('100', '150', '200', '300', '400', '600'))

    payment_mode = st.sidebar.selectbox('Payment Mode', ('1', '2', '4', '12'))

    payment_term = st.sidebar.slider('Payment Term', 0, 100)

    extra_load = st.sidebar.selectbox('Extra Load', ('Yes', 'No'))

    service_agent_edu_level = st.sidebar.selectbox('Service Agent Educational Level', ('H', 'P', 'S', 'T', 'U'))

    payment_method = st.sidebar.selectbox('Payment Method', ('B', 'C', 'D', 'F', 'R'))

    sell_agent_edu_level = st.sidebar.selectbox('Selling Agent Educational Level', ('H', 'P', 'S', 'T', 'U'))

    sell_agent_state = st.sidebar.selectbox('Selling Agent State',
                                            ('KEDAH', 'KELANTAN', 'JOHOR', 'KUALA LUMPUR', 'LABUAN', 'MELAKA',
                                             'NEGERI SEMBILAN', 'PAHANG', 'PERLIS', 'PENANG', 'PERAK', 'SABAH',
                                             'SARAWAK', 'SELANGOR', 'TERENGGANU'))

    sell_agent_age = st.sidebar.slider('Selling Agent Age', 0, 100)

    state = st.sidebar.selectbox('State',
                                 ('KEDAH', 'KELANTAN', 'JOHOR', 'KUALA LUMPUR', 'LABUAN', 'MELAKA', 'NEGERI SEMBILAN',
                                  'PAHANG', 'PERLIS', 'PENANG', 'PERAK', 'SABAH', 'SARAWAK', 'SELANGOR', 'TERENGGANU'))

    bmi = st.sidebar.number_input('BMI')

    data = {'RISK_CODE': risk_code, 'SEX': sex, 'OCCUPATION_CLASS': occupation_class, 'RACE': race,
            'MARITAL_STATUS': marital_status, 'ENTRY_AGE': entry_age, 'SMOKER': smoker, 'PAYMENT_MODE': payment_mode,
            'RSK_SUM_ASSURE': risk_sum_assured, 'PAYMNT_TERM': payment_term, 'EXTRA_LOAD': extra_load,
            'SERVICE_AGENT_EDU_LEVEL': service_agent_edu_level, 'PAYMENT_METHOD': payment_method,
            'SELL_AGENT_EDU_LEVEL': sell_agent_edu_level, 'SELL_AGENT_AGE': sell_agent_age,
            'SELL_AGENT_STATE': sell_agent_state, 'STATE': state, 'BMI': bmi
            }

    features = pd.DataFrame(data, index=[0])

    # Transform categorical features into the appropriate type that is expected by LightGBM
    for col in features.columns:
        col_type = features[col].dtype
        if col_type.name == 'object' or col_type.name == 'category':
            features[col] = features[col].astype('category')

    return features


df = userInputFeatures()

# Displays the user input features
st.subheader('User Input features')
st.write(df)

# Reads in saved classification model
load_clf = joblib.load(open(r'\\10.188.78.123\CP_Shared\lgbm_model_auc.pkl', 'rb'))

# read data from a file
# with open(r'\\10.188.78.123\CP_Shared\lgbm_model_auc.pkl', 'rb') as f:
#     # load_clf =
#     print(pickle.load(f))

print(df.info())
# st.write(df.columns)
# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

# Prediction
st.subheader('Prediction')
ecal = np.array(['Not Early Claim and Lapsed', 'Early Claim and Lapsed'])
st.write(ecal[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
