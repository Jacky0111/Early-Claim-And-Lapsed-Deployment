import joblib
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
from pathlib import Path
from DataManipulation import DataManipulation

col_list = ['RISK_CODE', 'SEX', 'OCCUPATION_CLASS', 'RACE', 'MARITAL_STATUS', 'ENTRY_AGE', 'SMOKER', 'PAYMENT_MODE',
            'RSK_SUM_ASSURE', 'PAYMNT_TERM', 'EXTRA_LOAD', 'SERVICE_AGENT_EDU_LEVEL', 'PAYMENT_METHOD',
            'SELL_AGENT_EDU_LEVEL', 'SELL_AGENT_AGE', 'SELL_AGENT_POSTCODE', 'STATE', 'BMI', 'POLICY_NO']

def policyNo():
    policyNum = st.sidebar.number_input('Policy Number', min_value=0,max_value=999999999999999, step=1)
    st.write("Policy Number :" + str(policyNum))
    
# Collects user input features into dataframe
def userInputFeatures():
    

    sell_agent_age = st.sidebar.slider('Selling Agent Age', 0, 100)

    payment_term = st.sidebar.slider('Payment Term', 0, 100)

    entry_age = st.sidebar.slider('Entry Age', 0, 100)

    bmi = st.sidebar.number_input('BMI')

    occupation_class = st.sidebar.selectbox('Occupation Class', ('A', 'B', 'C', 'D'))

    payment_method = st.sidebar.selectbox('Payment Method', ('B', 'C', 'D', 'F', 'R'))

    sell_agent_state = st.sidebar.selectbox('Selling Agent State',
                                            ('KEDAH', 'KELANTAN', 'JOHOR', 'KUALA LUMPUR', 'LABUAN', 'MELAKA',
                                             'NEGERI SEMBILAN', 'PAHANG', 'PERLIS', 'PENANG', 'PERAK', 'SABAH',
                                             'SARAWAK', 'SELANGOR', 'TERENGGANU'))

    state = st.sidebar.selectbox('State',
                                 ('KEDAH', 'KELANTAN', 'JOHOR', 'KUALA LUMPUR', 'LABUAN', 'MELAKA', 'NEGERI SEMBILAN',
                                  'PAHANG', 'PERLIS', 'PENANG', 'PERAK', 'SABAH', 'SARAWAK', 'SELANGOR', 'TERENGGANU'))

    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))

    risk_sum_assured = st.sidebar.selectbox('Risk Sum Assured', ('100', '150', '200', '300', '400', '600'))

    extra_load = st.sidebar.selectbox('Extra Load', ('Yes', 'No'))

    smoker = st.sidebar.selectbox('Smoker', ('Smoker', 'Non-Smoker'))

    payment_mode = st.sidebar.selectbox('Payment Mode', ('Monthly', 'Yearly', 'Half-yearly', 'Quarterly'))

    race = st.sidebar.selectbox('Race', ('BN', 'CH', 'IN', 'MA', 'OT'))

    risk_code = st.sidebar.selectbox('Risk Code',
                                     ('RTH1', 'RTI1', 'RTM1', 'RTN1', 'RTP1', 'RTP1', 'RTX1', 'UNHA', 'UNHC',
                                      'UNHD', 'UNHE', 'UNHG', 'UNHH', 'UNHJ', 'UNHK', 'UNHL', 'UNHP'))

    service_agent_edu_level = st.sidebar.selectbox('Service Agent Educational Level', ('H', 'P', 'S', 'T', 'U'))

    sell_agent_edu_level = st.sidebar.selectbox('Selling Agent Educational Level', ('H', 'P', 'S', 'T', 'U'))

    marital_status = st.sidebar.selectbox('Marital Status',
                                          ('Single', 'Married', 'Widowed', 'Divorced', 'Not Disclosed', 'Unknown'))

    data = {'RISK_CODE': risk_code, 'SEX': sex, 'OCCUPATION_CLASS': occupation_class, 'RACE': race,
            'MARITAL_STATUS': marital_status, 'ENTRY_AGE': entry_age, 'SMOKER': smoker, 'PAYMENT_MODE': payment_mode,
            'RSK_SUM_ASSURE': risk_sum_assured, 'PAYMNT_TERM': payment_term, 'EXTRA_LOAD': extra_load,
            'SERVICE_AGENT_EDU_LEVEL': service_agent_edu_level, 'PAYMENT_METHOD': payment_method,
            'SELL_AGENT_EDU_LEVEL': sell_agent_edu_level, 'SELL_AGENT_AGE': sell_agent_age,
            'SELL_AGENT_STATE': sell_agent_state, 'STATE': state, 'BMI': bmi
            }

    return pd.DataFrame(data, index=[0])


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def reindexDataFrame(df):
    columns_name = ['SELL_AGENT_AGE', 'PAYMNT_TERM', 'ENTRY_AGE', 'BMI', 'OCCUPATION_CLASS', 'PAYMENT_METHOD',
                    'SELL_AGENT_STATE', 'STATE', 'SEX', 'RSK_SUM_ASSURE', 'EXTRA_LOAD', 'SMOKER', 'PAYMENT_MODE',
                    'RACE', 'RISK_CODE', 'SERVICE_AGENT_EDU_LEVEL', 'SELL_AGENT_EDU_LEVEL', 'MARITAL_STATUS']
    return df.reindex(columns=columns_name)


def main():
    # Header
    st.write('# Early Claim and Lapsed Prediction')
    # Header of sidebar
    st.sidebar.header('User Input')

    target = pd.DataFrame()
    # Import file or user manually inputs
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        if Path(uploaded_file.name).suffix == '.xlsx':
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        # Drop useless columns
        df.drop([col for col in df.columns if col not in col_list], axis=1, inplace=True)

        # Drop duplicates
        df.drop_duplicates(inplace=True)
        df.reset_index(inplace=True, drop=True)

        pol_df = df[['POLICY_NO']].astype('category')
        df.drop('POLICY_NO', inplace=True, axis=1)

        # Clean Missing Values
        df = DataManipulation.missingValues(df)

        # Data Manipulation
        df['EXTRA_LOAD'] = DataManipulation.extraLoad(df['EXTRA_LOAD'])
        df['PAYMENT_MODE'] = DataManipulation.paymentMode(df['PAYMENT_MODE'])
        df['RSK_SUM_ASSURE'] = DataManipulation.riskSumAssured(df['RSK_SUM_ASSURE'])

        df['SELL_AGENT_POSTCODE'], drop_list = DataManipulation.agentPostcode(df['SELL_AGENT_POSTCODE'])
        df.drop(df['SELL_AGENT_POSTCODE'].index[drop_list], axis=0, inplace=True)
        pol_df.drop(pol_df['POLICY_NO'].index[drop_list], axis=0, inplace=True)

        df.reset_index(inplace=True, drop=True)
        pol_df.reset_index(inplace=True, drop=True)

        df.rename(columns={'SELL_AGENT_POSTCODE': 'SELL_AGENT_STATE'}, inplace=True)

    else:
        policyNo()
        df = userInputFeatures()

    df = DataManipulation.objToCat(df)

    # Reindex the sequence of the column
    dis_df = reindexDataFrame(df)

    # Displays the user input features
    st.subheader('User Input')
    st.write(dis_df)

    # # Reads in saved classification model
    #load_clf = joblib.load(open(r'\\10.188.78.123\CP_Shared\lgbm_model_auc.pkl', 'rb'))
    load_clf = joblib.load(open(r'C:\Users\User\OneDrive\Desktop\lgbm_model_auc.pkl', 'rb'))
    

    # Apply model to make predictions
    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)

    # Prediction
    st.subheader('Prediction')
    st.write('#### No : Not Early Claim and Lapsed')
    st.write('#### Yes: Early Claim and Lapsed')
    ecal = np.array(['Not Early Claim and Lapsed', 'Early Claim and Lapsed'])
    ecal_df = pd.DataFrame(ecal[prediction], columns=['Prediction'])

    proba_df = pd.DataFrame({'Predicted No (%)': prediction_proba[:, 0] * 100, 'Predicted Yes (%)': prediction_proba[:, 1] * 100})

    try:
        df2 = pd.concat([pol_df, ecal_df, proba_df], axis=1)
    except UnboundLocalError:
        df2 = pd.concat([ecal_df, proba_df], axis=1)
    df2 = df2.sort_values(by=['Predicted Yes (%)'], ascending=False)
    st.write(df2)

    try:
        df2['POLICY_NO'] = df2['POLICY_NO'].astype(str)
        df2['Predicted No (%)'] = df2['Predicted No (%)'].round(decimals=2)
        df2['Predicted Yes (%)'] = df2['Predicted Yes (%)'].round(decimals=2)

        df2_xlsx = to_excel(df2)
        st.download_button(label='Export Current Result',
                           data=df2_xlsx,
                           file_name=f'{Path(uploaded_file.name).stem}_result.xlsx')
    except KeyError:
        df2['Predicted No (%)'] = df2['Predicted No (%)'].round(decimals=2)
        df2['Predicted Yes (%)'] = df2['Predicted Yes (%)'].round(decimals=2)
        df2_xlsx = to_excel(df2)
        st.download_button(label='Export Current Result',
                           data=df2_xlsx,
                           file_name=f'result.xlsx')


if __name__ == '__main__':
    main()
