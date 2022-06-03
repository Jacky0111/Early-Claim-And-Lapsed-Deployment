import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from DataManipulation import DataManipulation

col_list = ['RISK_CODE', 'SEX', 'OCCUPATION_CLASS', 'RACE', 'MARITAL_STATUS', 'ENTRY_AGE', 'SMOKER', 'PAYMENT_MODE',
            'RSK_SUM_ASSURE', 'PAYMNT_TERM', 'EXTRA_LOAD', 'SERVICE_AGENT_EDU_LEVEL', 'PAYMENT_METHOD',
            'SELL_AGENT_EDU_LEVEL', 'SELL_AGENT_AGE', 'SELL_AGENT_POSTCODE', 'STATE', 'BMI']


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

    payment_mode = st.sidebar.selectbox('Payment Mode', ('Monthly', 'Yearly', 'Half-yearly', 'Quarterly'))

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

    return pd.DataFrame(data, index=[0])


def main():
    # Header
    st.write('# Early Claim and Lapsed Prediction')
    # Header of sidebar
    st.sidebar.header('User Input Features')

    target = pd.DataFrame()
    # Import file or user manually inputs
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        if Path(uploaded_file.name).suffix == '.xlsx':
            df = pd.read_excel(uploaded_file)
        #     uploaded_file.name = Path(uploaded_file.name).stem + '.csv'
        #     os.rename(uploaded_file.name, os.path.splitext(uploaded_file.name)[0] + '.xlsx')
        else:
            df = pd.read_csv(uploaded_file)

        # Store target column into new Dataframe
        target = df[['EarlyClaimAndLapsed']]

        # Column columns
        df.drop([col for col in df.columns if col not in col_list], axis=1, inplace=True)

        # Clean Missing Values
        df = DataManipulation.missingValues(df)

        # Data Manipulation
        df['EXTRA_LOAD'] = DataManipulation.extraLoad(df['EXTRA_LOAD'])
        df['PAYMENT_MODE'] = DataManipulation.paymentMode(df['PAYMENT_MODE'])
        df['RSK_SUM_ASSURE'] = DataManipulation.riskSumAssured(df['RSK_SUM_ASSURE'])

        df.reset_index(inplace=True, drop=True)
        df['SELL_AGENT_POSTCODE'] = DataManipulation.agentPostcode(df['SELL_AGENT_POSTCODE'])
        df.rename(columns={'SELL_AGENT_POSTCODE': 'SELL_AGENT_STATE'}, inplace=True)

    else:
        df = userInputFeatures()

    df = DataManipulation.objToCat(df)

    # Displays the user input features
    st.subheader('User Input features')
    st.write(df)

    # Reads in saved classification model
    load_clf = joblib.load(open(r'\\10.188.78.123\CP_Shared\lgbm_model_auc.pkl', 'rb'))

    # print(df.info())
    # Apply model to make predictions
    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)

    # Prediction
    st.subheader('Prediction')
    # ecal = np.array(['Not Early Claim and Lapsed', 'Early Claim and Lapsed'])
    ecal = np.array(['N', 'Y'])
    ecal_df = pd.DataFrame(ecal[prediction], columns=['Prediction'])
    df2 = pd.concat([target, ecal_df], axis=1)
    df2['Result'] = df2.apply(lambda x: True if x['EarlyClaimAndLapsed'] == x['Prediction'] else False, axis=1)
    st.write(df2)

    accuracy = df2.loc[df2['Result'] == True].shape[0] / df2.shape[0]
    st.write('####  ' + str(df2.loc[df2['Result'] == True].shape[0]) + ' out of ' + str(
        df2.shape[0]) + ' are predicted correct, ' +
             'where the accuracy is ' + str(round(accuracy, 2) * 100) + '%.')

    st.subheader('Prediction Probability')
    st.write(prediction_proba)


if __name__ == '__main__':
    main()
