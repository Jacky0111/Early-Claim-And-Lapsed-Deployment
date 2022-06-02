
    # In[1]:


    # Import Libraries and Packages
def preprocessing(filename):
    # Data processing
    import re
    # import missingno
    import pandas as pd
    import numpy as np
    # pd.set_option("display.max_rows", None, "display.max_columns", None)

    # Visualization
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['figure.figsize'] = (25, 15)
    # get_ipython().run_line_magic('matplotlib', 'inline')
    import plotly
    import plotly.express as px 
    import plotly.io as pio 
    import plotly.offline as py
    py.init_notebook_mode()
    import plotly.graph_objs as go 
    import plotly.tools as tls 
    import plotly.figure_factory as ff 
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    pyo.init_notebook_mode()
    from IPython.display import Markdown

    # Feature Engineering
    from imblearn.over_sampling import SMOTE, SMOTENC, ADASYN
    from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder #, OrdinalEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline, Pipeline # Pipeline
    from scipy.stats import pearsonr

    # Model Training
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, cross_val_score, KFold, StratifiedKFold
    from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, average_precision_score
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, auc
    import lightgbm as lgbm
    from lightgbm import LGBMClassifier

    # Model Optimization
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances


    # In[2]:


    # Read Merged dataset
    claim_df = pd.read_excel(filename)
    claim_df


    # In[3]:


    # Check whether there are whitespace value in the dataframe
    claim_df.iloc[:, 32].values[0].isspace()


    # In[4]:


    # # First Solution
    # def convert_to_nan(x):
    #     if str(x).isspace():
    #         return np.NaN
    #     elif x == None:
    #         return np.NaN
    #     else:
    #         return str(x).strip()
    # claim_df.applymap(convert_to_nan)

    # Second Solution
    # (28, 32, 40, 46, 49, 57, 59, 63, 64, 65, 67)
    for col in range(claim_df.shape[1]):
        if claim_df.iloc[:, col].dtype == 'object':
            for index, val in enumerate(claim_df.iloc[:, col].values):
                claim_df.iloc[:, col].values[index] = str(val).strip()
                if str(val).isspace() or val is None or str(val).lower() == 'nan' or str(val).lower() == 'none' or val == '':
                    claim_df.iloc[:, col].values[index] = np.NaN

    # # Third Solution
    # # Strip all the front and end whitespace
    # for column in claim_df.columns:
    #     claim_df[column] = claim_df[column].apply(lambda x: str(x).strip())

    # # Replace empty string ('' or "") with np.NAN value
    # for column in claim_df.columns:
    #     claim_df[column] = claim_df[column].apply(lambda x: str(x).replace('', str(np.NaN)))
    #     claim_df[column] = np.where(claim_df[column].values == 'nan', np.NaN, claim_df[column])

    # # Replace None with np.NAN value
    # for column in claim_df.columns:
    #     claim_df[column] = claim_df[column].apply(lambda x: str(x).lower().replace('none', str(np.NaN)))
    #     claim_df[column] = np.where(claim_df[column].values == 'none', np.NaN, claim_df[column])

    # # Replace None with np.NAN value
    # for column in claim_df.columns:
    #     claim_df[column] = claim_df[column].apply(lambda x: str(x).lower().replace('nan', str(np.NaN)))
    #     claim_df[column] = np.where(claim_df[column].values == 'nan', np.NaN, claim_df[column])
        
    # Display the result
    claim_df


    # In[5]:


    # Check whether the value is whitespace again
    try:
        claim_df.iloc[:, 32].values[0].isspace()
    except AttributeError:
        print(f'The value now becomes {claim_df.iloc[:, 32].values[0]}. Is it not whitespace anymore')

    import math
    print(f"Check if it is np.NaN: {math.isnan(claim_df.iloc[:, 32].values[0])}")


    # In[6]:


    # Columns that have mixed type problem
    # display(Markdown(f"#### Columns that have mixed type problem:"))
    for index in (28, 32, 40, 46, 49, 57, 59, 63, 64, 65, 67):
        print(f"Column Index: {index}\nColumn Name: {claim_df.iloc[:, index].name}\n")


    # In[7]:


    # Make sure these 2 columns are in float data type
    claim_df['SELL_AGENT_POSTCODE'] = claim_df['SELL_AGENT_POSTCODE'].astype(float)
    claim_df['SERVICE_AGENT_POSTCODE'] = claim_df['SERVICE_AGENT_POSTCODE'].astype(float)


    # In[8]:


    # Convert those columns that have mixed typed of data
    claim_df.iloc[:, 28] = claim_df.iloc[:, 28].apply(str)
    claim_df.iloc[:, 32] = claim_df.iloc[:, 32].astype(float)
    # claim_df.iloc[:, 40] = pd.to_datetime(claim_df.iloc[:, 40], dayfirst=True)
    # claim_df.iloc[:, 46] = pd.to_datetime(claim_df.iloc[:, 46], dayfirst=True)
    # claim_df.iloc[:, 49] = pd.to_datetime(claim_df.iloc[:, 49], dayfirst=True)
    # claim_df.iloc[:, 57] = pd.to_datetime(claim_df.iloc[:, 57], dayfirst=True)
    # claim_df.iloc[:, 59] = pd.to_datetime(claim_df.iloc[:, 59], dayfirst=True)
    claim_df.iloc[:, 63] = claim_df.iloc[:, 63].astype(float)
    claim_df.iloc[:, 64] = claim_df.iloc[:, 64].astype(float)
    claim_df.iloc[:, 65] = claim_df.iloc[:, 65].astype(float)
    claim_df.iloc[:, 67] = claim_df.iloc[:, 67].apply(str)

    for x in claim_df.columns.tolist():
        if 'nan' in claim_df[x].value_counts().keys() or 'none' in claim_df[x].value_counts().keys():
            for index, val in enumerate(claim_df[x].values):
                if val == 'nan' or val == 'none':
                    claim_df[x].values[index] = np.NaN


    # In[9]:


    print(f"\nInsurance claims dataset shape")
    print("-----------------------------------------")
    print("Rows     : " , claim_df.shape[0])
    print("Columns  : " , claim_df.shape[1])
    print("Shape    : " , claim_df.shape)
    print("Features : \n" , claim_df.columns.tolist())


    # Based on our study with the staff of Analytics & Strategy Department, there are some **time durations we should take into consideration**, the time durations that we should take into account are as follow:
    # 
    # 1. Duration between **POL_COMMENCE_DATE & DOA**
    # 
    # Therefore, we calculate the above durations in **days** and add them as features to our dataset.
    # 
    # |Columns   |Explainations   |
    # |:---|:---|
    # |**DOA**   |Date of hospital admission   |
    # |**POL_COMMENCE_DATE**   |Policy effective start date   |

    # In[10]:


    # Since we load the data from CSV file, pandas will treat date columns as string/object, so we are required 
    # convert them back to date
    claim_df['DOA'] = pd.to_datetime(claim_df['DOA'], dayfirst=True)

    claim_df['POL_COMMENCE_DATE'] = pd.to_datetime(claim_df['POL_COMMENCE_DATE'], dayfirst=True)

    # claim_df['POL_STATUS_DATE'] = pd.to_datetime(claim_df['POL_STATUS_DATE'], dayfirst=True)

    # claim_df['PAID_TO_DATE'] = pd.to_datetime(claim_df['PAID_TO_DATE'], dayfirst=True)

    # View the info after changing to datetime64[ns] data type
    claim_df[['DOA', 'POL_COMMENCE_DATE']].info()


    # In[11]:


    # View some of the samples above after done converting them to date
    claim_df[['DOA', 'POL_COMMENCE_DATE']].head()


    # In[12]:


    # Gap between Policy Commencement Date and DOA.
    claim_df['GAP_BET_DOA_PCD'] = claim_df.apply(lambda row: (row['DOA'] - row['POL_COMMENCE_DATE']).days, axis=1)


    # In[13]:


    # Calculate whether the days in GAP_BET_DOA_PCD is within 2 years, replace with 1 if yes, otherwise 0
    # 1 indicating within 2 years
    # 0 indicating not within 2 years
    claim_df['GAP_BET_DOA_PCD'] = claim_df['GAP_BET_DOA_PCD'].apply(lambda x: 1 if x <= 730 else 0)
    claim_df.rename(columns = {'GAP_BET_DOA_PCD':'IS_POL_IN_2YEARS'}, inplace=True)
    claim_df['IS_POL_IN_2YEARS'] = claim_df['IS_POL_IN_2YEARS'].apply(str)

    # View some of the samples above after done converting them to date
    claim_df[['IS_POL_IN_2YEARS']]


    # #### Select those policy data that is within 2 years timeframe

    # In[14]:


    claim_df = claim_df.loc[claim_df['IS_POL_IN_2YEARS'] == '1']
    claim_df['IS_POL_IN_2YEARS'].value_counts()


    # In[15]:


    # Drop the 'IS_POL_IN_2YEARS' column bcus it is not useful anymore
    claim_df.drop(labels=['IS_POL_IN_2YEARS'], axis=1, inplace=True)

    # Move the "EarlyClaimAndLapsed" column to the last
    last_column = claim_df.pop('EarlyClaimAndLapsed')
    claim_df.insert(claim_df.shape[1], 'EarlyClaimAndLapsed', last_column) 

    # Display data
    claim_df


    # In[16]:


    # Check if there any duplicate rows left
    claim_df[claim_df.duplicated()].shape[0]


    # **After going through a study on every feature we have, we have filtered out those features that are likely to not to include in the dataset for the classification task. Below are the features (a.k.a. columns) that we are going to drop.**

    # In[17]:


    year_date_cols = ['POL_ISSUE_YEAR', 'POL_COMMENCE_DATE', 'EXP_DATE', 'DOA', 'LAST_PAYMT_DATE', 
                    'RSK_COMMENCE_DATE', 'RSK_STATUS_DATE', 'POL_LAPSE_DATE', 'POL_STATUS_DATE', 'FIRST_ISSUE_DATE', 
                    'PAID_TO_DATE', 'CLAIM_RECOD_DATE', 'FLAT_RATE_LOAD_YEAR', 'POL_ISSUE_DATE', 'POLICY_YEAR', 
                    'SERVICE_AGENT_JOIN_DATE', 'SELL_AGENT_JOIN_DATE']

    # agent_cols = ['SERVICE_AGENT_NO', 'SERVICE_AGENT_SEX', 'SERVICE_AGENT_RACE', 'SERVICE_AGENT_BRANCH_CODE', 
    #               'SELL_AGENT_NO', 'SELL_AGENT_SEX', 'SELL_AGENT_RACE', 'SELL_AGENT_BRANCH_CODE']

    agent_cols = ['SERVICE_AGENT_NO', 'SERVICE_AGENT_BRANCH_CODE', 'SELL_AGENT_NO', 'SELL_AGENT_BRANCH_CODE', 
                'SELL_AGENT_RACE', 'SELL_AGENT_SEX', 'SERVICE_AGENT_RACE', 'SERVICE_AGENT_SEX']

    payment_cols = [] # 'PAYMNT_TERM'


    # In[18]:


    '''
    SINGLE_PREMIUM: 74260 of 74267 rows are blank. More than half (99.99%) of the data are blank.
    TOPUP_PREMIUM: 74260 of 74267 rows are blank. More than half (99.99%) of the data are blank.
    EXCESS_PREMIUM: All blank (100%).
    INDCATE_FLEX_PREM: 71483 of 74267 rows are blank. More than half (96.25%) of the data are blank.
    ANUAL_PREMIUM: 46395 of 74267 rows are blank. More than half (62.47%) of the data are blank.
    ANUAL_EX_PREMIUM: 46395 of 74267 rows are blank. More than half (62.47%) of the data are blank.
    PREM_STATUS: No blank.
    '''
    prem_cols = ['SINGLE_PREMIUM', 'TOPUP_PREMIUM', 'EXCESS_PREMIUM', 'INDCATE_FLEX_PREM', 'ANUAL_PREMIUM', 'ANUAL_EX_PREMIUM', 
                'PREM_STATUS']

    # import warnings
    # warnings.filterwarnings('ignore')
    # sweet_report = sv.analyze(source=claim_df[prem_cols + ['EarlyClaimAndLapsed']], target_feat='EarlyClaimAndLapsed')
    # sweet_report.show_notebook(w="100%", h="Full", layout='vertical')


    # In[19]:


    '''
    CURRENCY_CODE: All rows only have 1 category value only which is 'RM'.

    MATURE_AGE: Range from 13 to 90.

    PAR_INDICATE: 46395 of 74267 rows are blank. More than half (62.47%) of the data are blank. 
                14117 of 74267 rows are 'P2' valued. 
                9781 of 74267 rows are 'P1' valued.
                3974 of 74267 rows are 'NP' valued.
                
    FACULATIVE_IND: 27872 of 74267 rows are blank. More than half 1/4 (37.53%) of the data are blank.
                    Only 1 category value only which is 'N' (62.47%).
                    An indicator of whether the policy is under reinsurance facultative agreement. Advised to drop this field
                    
    FLAT_RATE_LOAD: 46395 of 74267 rows are blank. More than half (62.47%) of the data are blank.
                    Only 1 category value only which is 0 (37.26%). 

    LIFE_COVERAGE: Range from 1 to 2

    COVER_NO: Do not include bcus don't have 2019-2021 data. Advised to drop this field

    RISK_TERM: 

    RIDER_NO: Do not include bcus don't have 2019-2021 data. Advised to drop this field
    '''

    other_cols = ['POLICY_NO', 'BRANCH_CODE', 'CURRENCY_CODE', 'MATURE_AGE', 'PAR_INDICATE', 'FACULATIVE_IND', 
                'FLAT_RATE_LOAD', 'LIFE_COVERAGE', 'COVER_NO', 'RISK_TERM', 'POLICY_STATUS', 'RIDER_NO', 
                'RISK_STATUS', 'RISK_STATUS.1'] # 'BRANCH_CODE', 'SEX', 'RACE'

    # sweet_report = sv.analyze(source=claim_df[other_cols + ['EarlyClaimAndLapsed']], target_feat='EarlyClaimAndLapsed')
    # sweet_report.show_notebook(w="100%", h="Full", layout='vertical')


    # In[20]:


    total_drop_cols = len(year_date_cols + agent_cols + payment_cols + other_cols + prem_cols)
    print(f'Total Columns to be Dropped: {total_drop_cols} out of {claim_df.shape[1]}')
    print(f'Total Columns Left: {claim_df.shape[1] - total_drop_cols}')

    # Drop useless columns
    claim_df.drop(columns=year_date_cols + agent_cols + payment_cols + other_cols + prem_cols, inplace=True, axis=1)
    claim_df.drop_duplicates(inplace=True)


    # In[21]:


    # Check again if there any duplicate rows left
    claim_df[claim_df.duplicated()].shape[0]


    # In[22]:


    claim_df.dropna(subset=['SELL_AGENT_POSTCODE', 'SERVICE_AGENT_POSTCODE'], inplace=True)
    claim_df.reset_index(inplace=True, drop=True)


    # <u>**Source:**</u>
    # 1. [A Website to check Malaysia Postcode](http://www.poskod.com/index.html)

    # In[23]:


    def mapPostCode(claim_df):
        i = 0
        while i < claim_df.count():        
            if claim_df[i] in range (50000,60001) :
                claim_df[i] = 'KUALA LUMPUR'
            elif str(claim_df[i]).startswith('62') :
                claim_df[i] = 'PUTRAJAYA'
            elif str(claim_df[i]).startswith('87') :
                claim_df[i] = 'LABUAN'
            elif claim_df[i] in range (1000,2801) :
                claim_df[i] = 'PERLIS'
            elif claim_df[i] in range (5000,9811) or claim_df[i] == 14290 or claim_df[i] == 14390 or claim_df[i] == 34950:
                claim_df[i] = 'KEDAH'
            elif claim_df[i] in range (10000,14401) :
                claim_df[i] = 'PENANG'
            elif claim_df[i] in range (15000,18501) :
                claim_df[i] = 'KELANTAN'
            elif claim_df[i] in range (20000,24301) :
                claim_df[i] = 'TERENGGANU'
            elif claim_df[i] in range (25000,28801) or str(claim_df[i]).startswith('39') or str(claim_df[i]).startswith('49') or claim_df[i] == 69000:
                claim_df[i] = 'PAHANG'
            elif claim_df[i] in range (30000,36811) :
                claim_df[i] = 'PERAK'
            elif claim_df[i] in range (40000,48301) or claim_df[i] in range (63000,63301) or claim_df[i] == 64000 or claim_df[i] in range (68000,68101):
                claim_df[i] = 'SELANGOR'  
            elif claim_df[i] in range (70000,73501) :
                claim_df[i] = 'NEGERI SEMBILAN'
            elif claim_df[i] in range (75000,78301) :
                claim_df[i] = 'MELAKA'
            elif claim_df[i] in range (79000,86901) :
                claim_df[i] = 'JOHOR'
            elif claim_df[i] in range (88000,91301) :
                claim_df[i] = 'SABAH'
            elif claim_df[i] in range (93000,98851) :
                claim_df[i] = 'SARAWAK'
            i+=1

    mapPostCode(claim_df['SELL_AGENT_POSTCODE'])
    mapPostCode(claim_df['SERVICE_AGENT_POSTCODE'])
    claim_df


    # In[24]:


    import numbers

    claim_df.rename(columns = {'SELL_AGENT_POSTCODE':'SELL_AGENT_STATE', 'SERVICE_AGENT_POSTCODE':'SERVICE_AGENT_STATE'}, 
                    inplace = True)

    fail_con_count = {}
    # display(Markdown(f"#### Total number rows: {claim_df.shape[0]}"))
    for agnt_postcode in ['SELL_AGENT_STATE', 'SERVICE_AGENT_STATE']:
        count = 0
        for x in claim_df[agnt_postcode].values:
            if isinstance(x, numbers.Number) == True:
                count = count + 1
        fail_con_count[agnt_postcode] = count
        print(f"Column Name: {agnt_postcode}")
        print(f"Number of postcodes that fail to convert to state: {count}")


    # In[25]:


    if fail_con_count['SELL_AGENT_STATE'] != 0 or fail_con_count['SERVICE_AGENT_STATE'] != 0:
        for col in ['SELL_AGENT_STATE', 'SERVICE_AGENT_STATE']:
            for poskod in claim_df[col]:
                if isinstance(poskod, numbers.Number) == True:
                    print(f'Postcode {poskod} does not exist in Malaysia')
                    claim_df = claim_df.loc[claim_df[col] != poskod]


    # In[26]:


    # display(Markdown(f"#### Total number rows: {claim_df.shape[0]}"))
    for agnt_postcode in ['SELL_AGENT_STATE', 'SERVICE_AGENT_STATE']:
        count = 0
        for x in claim_df[agnt_postcode].values:
            if isinstance(x, numbers.Number) == True:
                count = count + 1
        print(f"Column Name: {agnt_postcode}")
        print(f"Number of postcodes that fail to convert to state: {count}")


    # In[27]:


    for x in claim_df.columns.tolist():
        if 'none' in claim_df[x].value_counts().keys():
            print(x)


    # In[28]:


    # Check again if there any duplicate rows left
    print(f"Any duplicate rows left?\nAnswer: {claim_df[claim_df.duplicated()].shape[0]}\n")

    # If there any duplicate rows left, drop it
    if claim_df[claim_df.duplicated()].shape[0] > 0:
        print("\nDuplicate rows will be cleared now...")
        claim_df.drop_duplicates(inplace=True)
        print(f"Any duplicate rows left?\nAnswer: {claim_df[claim_df.duplicated()].shape[0]}")


    # ## 2.2 Explonary Data Analysis (EDA)

    # ### 2.2.1 Data Shape

    # In[29]:


    print(f"\nMerged (claim_df) dataset shape")
    print("---------------------------------")
    print ("Rows     : " , claim_df.shape[0])
    print ("Columns  : " , claim_df.shape[1])
    print ("Shape    : " , claim_df.shape)
    print ("Features : \n" , claim_df.columns.tolist())


    # ### 2.2.2 Data Info

    # In[30]:


    claim_df.info()
    # print(claim_df.iloc[:, :50].info())
    # print()
    # print(claim_df.iloc[:, 50:].info())


    # ### 2.2.3 Data Unique Values (Categorical Features)

    # In[31]:


    # Reset the index of dataframe so that the index values can be moved out as the first column in the dataframe.
    # display(Markdown(f"#### Unique values of each categorical features:"))
    pd.DataFrame(claim_df[claim_df.select_dtypes(include = ['object']).keys().tolist()].nunique(), 
                columns=['Number of Unique Values']).reset_index().rename(columns = {'index':'Categorical Features'}).sort_values(by=['Number of Unique Values'], ascending = False)


    # ### 2.2.4 Data Summary Statistic

    # In[32]:


    # Description of the dataset
    claim_df.describe()


    # ### 2.2.5 Understanding Data Distribution

    # In[33]:


    pie_display = claim_df

    # labels
    lab = pie_display["EarlyClaimAndLapsed"].value_counts().keys().tolist()

    # values
    val = claim_df["EarlyClaimAndLapsed"].value_counts().values.tolist()

    trace = go.Pie(labels = lab , values = val, marker = dict(colors = ['blue' ,'green'], line = dict(color = "white", width =  1.3)), rotation = 90, hoverinfo = "label + value + text", hole = 0.5)
    layout = go.Layout(dict(title = "Distribution of Early & Lapsed Claimed in Data", title_x = 0.5, plot_bgcolor  = "rgb(243, 243, 243)", paper_bgcolor = "rgb(243, 243, 243)",))

    import plotly.io as pio
    pio.renderers.default='notebook'

    fig = go.Figure(data = [trace], layout = layout)
    py.iplot(fig)


    # ### 2.2.6 Understanding Missing Values

    # In[34]:


    # Detect any missing value from dataset
    count = claim_df.isnull().sum().sort_values(ascending = False)
    percentage = ((claim_df.isnull().sum() / len(claim_df) * 100).sort_values(ascending = False))
    data_missing = pd.concat([count, percentage], axis = 1, keys = ['Count', 'Percentage (%)'])
    data_missing = data_missing.loc[data_missing['Count'] != 0]

    # display(Markdown(f"#### Count and Percentage of missing values for the columns:"))
    data_missing = data_missing.reset_index()
    data_missing.rename(columns = {'index':'Features'}, inplace = True)
    data_missing


    # In[35]:


    # display(Markdown(f"#### Total missing values: {data_missing.Count.sum()}"))
    # plot_missing(df=claim_df)


    # #### Data Manipulation
    # After knowing that the **`EXTRA_LOAD`** column in our dataset has quite many missing values, we decide to **convert those missing data to value 0** and **those that are not missing data to 1** as **`EXTRA_LOAD`** column is the amount of additional loading that Tokio Marine imposes to the customers for the higher risk customer like customer who has heart attack previously, high blood pressure, pre-existing condition like diabetes and etc.
    # 
    # Meanwhile, we will drop the **`SUBSTD_FLAG`** column as it has too many missing values.
    # 
    # <u><b>Source:</b></u>
    # 1. [How to replace values using apply() function?](https://stackoverflow.com/questions/63544400/how-to-replace-values-using-apply-function)

    # In[36]:


    backup_df = claim_df.copy()
    backup_df

    # claim_df = backup_df.copy()


    # In[37]:


    # Convert the missing value in EXTRA_LOAD field with 1 and others value with 0
    claim_df['EXTRA_LOAD'][:] = claim_df['EXTRA_LOAD'].apply(lambda x: 0 if math.isnan(x)==True else (0 if x==0 else 1))
    claim_df['EXTRA_LOAD'] = claim_df['EXTRA_LOAD'].astype(int).astype(str)
    claim_df['EXTRA_LOAD'].value_counts()


    # In[38]:


    # Drop 'SUBSTD_FLAG' column
    claim_df.drop(labels=['SUBSTD_FLAG'], inplace=True, axis=1)


    # In[39]:


    # View the missing values again
    count = claim_df.isnull().sum().sort_values(ascending = False)
    percentage = ((claim_df.isnull().sum() / len(claim_df) * 100).sort_values(ascending = False))
    data_missing = pd.concat([count, percentage], axis = 1, keys = ['Count', 'Percentage (%)'])
    data_missing = data_missing.loc[data_missing['Count'] != 0]

    # display(Markdown(f"#### Count and Percentage of missing values for the columns:"))
    data_missing = data_missing.reset_index()
    data_missing.rename(columns = {'index':'Features'}, inplace = True)
    data_missing


    # In[40]:


    # View the otal missing values again
    # display(Markdown(f"#### Total missing values: {data_missing.Count.sum()}"))


    # Then, we will proceed with **dropping all the rows that contain missing values**. Meanwhile, we also do some manupulations on the dataset after going through some observations and studies on the dataset in the excel file.

    # In[41]:


    # Drop all rows containing NaN values
    claim_df.dropna(inplace=True)

    # Shape of dataset after dropping NaN values
    print('Shape of dataset after dropping NaN values:')
    claim_df.shape


    # In[42]:


    '''
    In the STATE column, there may be some areas that are not part of Malaysia, so we need to delete these 
    areas and select the areas that are part of Malaysia.
    '''
    claim_df = claim_df.loc[(claim_df['STATE'] == 'JOHOR') | (claim_df['STATE'] == 'KEDAH') | (claim_df['STATE'] == 'KELANTAN') 
                            | (claim_df['STATE'] == 'KUALA LUMPUR') | (claim_df['STATE'] == 'LABUAN') 
                            | (claim_df['STATE'] == 'MELAKA') | (claim_df['STATE'] == 'NEGERI SEMBILAN') 
                            | (claim_df['STATE'] == 'PAHANG') | (claim_df['STATE'] == 'PENANG') | (claim_df['STATE'] == 'PERAK') 
                            | (claim_df['STATE'] == 'PERLIS') | (claim_df['STATE'] == 'SABAH') 
                            | (claim_df['STATE'] == 'SARAWAK') | (claim_df['STATE'] == 'SELANGOR') 
                            | (claim_df['STATE'] == 'TERENGGANU') | (claim_df['STATE'] == 'PUTRAJAYA')]

    # Replace the proper values for PAYMENT_MODE and convert the data typpe from float to object
    '''1 -  Annual, 2: Half-Yearly, 4: Quarterly, 12: Monthly'''
    claim_df['PAYMENT_MODE'] = claim_df['PAYMENT_MODE'].replace({12.: 'Monthly', 1.: 'Yearly', 
                                                                2.: 'Half-Yearly', 4.: 'Quarterly'}).astype(str)
    '''
    Convert the data type from float to object, because the values of RSK_SUM_ASSURE are not represented to the price 
    but for the packages that the policyholders bought
    '''
    claim_df['RSK_SUM_ASSURE'] = claim_df['RSK_SUM_ASSURE'].astype(int).astype(str)


    # Use **info()** function from `DataFrame` to view the data info again after some data manipulations

    # In[43]:


    claim_df.info()


    # In[44]:


    # display(Markdown(f"#### Total missing values: {claim_df.isnull().sum().sum()}"))


    # In[45]:


    # Check duplicate rows again
    if claim_df[claim_df.duplicated()].shape[0] > 0:
        print(f"There are {claim_df[claim_df.duplicated()].shape[0]} duplicated rows.")
        print("\nDuplicate rows will be cleared now...")
        claim_df.drop_duplicates(inplace=True)
    else:
        print("Congratz! No duplicate rows.")


    # ### 2.2.7 Overall Insights of EDA using AutoEDA

    # In[46]:


    # from dataprep.eda import plot
    # plot(claim_df)


    # In[47]:


    # Using sweetwiz to generate report
    # import warnings
    # warnings.filterwarnings('ignore')

    # sweet_report = sv.analyze(source=claim_df, target_feat='EarlyClaimAndLapsed')
    # sweet_report.show_notebook(w="100%", h="Full", layout='vertical')


    # ### 2.2.8 Categorical Variable vs EarlyClaimAndLapsed
    # 
    # <u>**Source**</u>
    # 1. [Adding percentage of count to a stacked bar chart in plotly](https://stackoverflow.com/questions/65233123/adding-percentage-of-count-to-a-stacked-bar-chart-in-plotly)

    # In[48]:


    for cat_col in claim_df.select_dtypes(include = ['object']).keys().tolist():
        
        if cat_col != 'EarlyClaimAndLapsed':
            df_g = claim_df[[cat_col, 'EarlyClaimAndLapsed']].groupby([cat_col, 'EarlyClaimAndLapsed']).size().reset_index()
            df_g['percentage'] = claim_df[[cat_col, 'EarlyClaimAndLapsed']].groupby([cat_col, 'EarlyClaimAndLapsed']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = [cat_col, 'EarlyClaimAndLapsed', 'Counts', 'Percentage']
            df_g = df_g.sort_values(['Counts'], ascending=False).reset_index(drop=True)
            
            fig = px.bar(df_g, x=cat_col, y=['Counts'], color='EarlyClaimAndLapsed', text=df_g['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)), barmode='group')
            fig.update_layout(title_text=f'{cat_col} vs EarlyClaimAndLapsed', title_x = 0.5, font_size=12)
            fig.show()
            
            # fig = px.histogram(claim_df, x=cat_col, y="EarlyClaimAndLapsed", color='EarlyClaimAndLapsed', 
            #                    barmode='group', histfunc='count', height=400, text_auto=True)
            # fig.update_layout(title_text=f'{cat_col} vs EarlyClaimAndLapsed', title_x = 0.5, font_size=12)
            # fig.show()


    # ### 2.2.9 Continuous Variable vs EarlyClaimAndLapsed
    # <u>**Source**</u>
    # 1. [Overlaying two histograms with plotly express](https://stackoverflow.com/questions/57988604/overlaying-two-histograms-with-plotly-express)
    # 2. [create a histogram with plotly.graph_objs like in plotly.express](https://stackoverflow.com/questions/71623896/create-a-histogram-with-plotly-graph-objs-like-in-plotly-express)
    # 3. [Plotly graph_objects add df column to hovertemplate](https://stackoverflow.com/questions/62825101/plotly-graph-objects-add-df-column-to-hovertemplate)

    # In[49]:


    # for con_col in claim_df.select_dtypes(include = ['number']).keys().tolist():
    #     df_g = claim_df[[con_col, 'EarlyClaimAndLapsed']].groupby([con_col, 'EarlyClaimAndLapsed']).size().reset_index()
    #     df_g['Percentage'] = claim_df[[con_col, 'EarlyClaimAndLapsed']].groupby([con_col, 'EarlyClaimAndLapsed']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
    #     df_g.columns = [con_col, 'EarlyClaimAndLapsed', 'Counts', 'Percentage']
        
    #     def find_percent(row_in_df):
    #         for index, row in df_g.iterrows():
    #             if row.tolist()[0:2] == row_in_df.tolist():
    #                 return row.tolist()[-1]

    #     df = claim_df[[con_col, 'EarlyClaimAndLapsed']].copy()
    #     df['Percentage'] = df.apply(find_percent, axis=1)
            
    #     fig = px.histogram(df, x=con_col, color='EarlyClaimAndLapsed', histfunc='count', barmode='overlay', hover_data=["Percentage"])
    #     fig.update_layout(title_text=f'{con_col} vs EarlyClaimAndLapsed', title_x = 0.5, font_size=12)
    #     fig.show()


    # In[50]:


    lapsed_df = claim_df[claim_df['EarlyClaimAndLapsed']=='Y']
    non_lapsed_df = claim_df[claim_df['EarlyClaimAndLapsed']=='N']

    for con_col in claim_df.select_dtypes(include = ['number']).keys().tolist():

        fig = go.Figure()
        fig.add_trace(go.Histogram(x = non_lapsed_df[con_col], name='N', marker_color='blue', hovertext=non_lapsed_df['EarlyClaimAndLapsed']))
        fig.add_trace(go.Histogram(x = lapsed_df[con_col], name='Y', marker_color='red', hovertext=lapsed_df['EarlyClaimAndLapsed']))
        fig.update_layout(title=f'{con_col} vs EarlyClaimAndLapsed', title_x=0.5, barmode="overlay", 
                        legend_title='EarlyClaimAndLapsed', xaxis_title=f'{con_col}', yaxis_title='Count'
                        )
        fig.update_traces(opacity=0.5, nbinsx=10, histfunc='count', hovertemplate="<br>".join([
                f"{con_col}"+": %{x}",
                "Count: %{y}", 
                "EarlyClaimAndLapsed: %{hovertext}"
        ]))
        fig.show()


    # In[51]:


    for con_col in claim_df.select_dtypes(include = ['number']).keys().tolist():
        fig = px.box(claim_df, y=con_col, color='EarlyClaimAndLapsed', )
        fig.update_layout(title_text=f'{con_col} vs EarlyClaimAndLapsed', title_x = 0.5, font_size=12)
        fig.show()


    # In[52]:


    early = None
    not_early = None
    target_col = []
    cat_cols = []
    num_cols = []
    bin_cols = []
    multi_cols = []
    ord_cols = []
    nom_cols = []

    def update_category_numeric_cols():
        # Separating early and non early customers
        early_fn = claim_df[claim_df['EarlyClaimAndLapsed'] == 'Y']
        not_fn_early = claim_df[claim_df['EarlyClaimAndLapsed'] == 'N']

        # Separating catagorical and numerical columns

        ## target features
        target_col.clear()
        target_col.append('EarlyClaimAndLapsed')

        ## Category Columns
        cat_fn_cols = claim_df.select_dtypes(include = ['object']).keys().tolist()
        cat_fn_cols.remove("EarlyClaimAndLapsed")
        cat_cols.clear()
        for x in cat_fn_cols:
            cat_cols.append(x)

        ## Numeric columns
        num_fn_cols = [x for x in claim_df.columns if x not in cat_cols + target_col and (claim_df[x].dtypes == np.float64 or claim_df[x].dtypes == np.int64)]
        num_cols.clear()
        for x in num_fn_cols:
            num_cols.append(x)
        
        ## Binary columns with 2 values
        bin_fn_cols = claim_df.select_dtypes(include = ['object']).nunique()[claim_df.select_dtypes(include = ['object']).nunique() == 2].keys().tolist()
        bin_fn_cols.remove("EarlyClaimAndLapsed")
        bin_cols.clear()
        for x in bin_fn_cols:
            bin_cols.append(x)

        ## Columns more than 2 values
        multi_fn_cols = [x for x in cat_cols if x not in bin_cols]
        multi_cols.clear()
        for x in multi_fn_cols:
            multi_cols.append(x)

        ## Data that have a predetermined/natural order or rank
        if 'OCCUPATION_CLASS' in claim_df.columns.tolist() and 'RSK_SUM_ASSURE' in claim_df.columns.tolist():
            ord_fn_cols = ['OCCUPATION_CLASS', 'RSK_SUM_ASSURE']
        elif 'OCCUPATION_CLASS' in claim_df.columns.tolist():
            ord_fn_cols = ['OCCUPATION_CLASS']
        elif 'RSK_SUM_ASSURE' in claim_df.columns.tolist():
            ord_fn_cols = ['RSK_SUM_ASSURE']
        else:
            ord_fn_cols = []
        ord_cols.clear()
        for x in ord_fn_cols:
            ord_cols.append(x)

        ## Data that are classfied without ranking
        nom_fn_cols = [x for x in cat_cols if x not in ord_cols]
        nom_cols.clear()
        for x in nom_fn_cols:
            nom_cols.append(x)

        ## Create a dataframe to view all features type
        feature_type = ['Target (Binary Categorical)', 'Categorical (Exclude Target)', 'Numeric', 'Binary Categorical (Exclude Target)', 'Multi-Class Categorical', 'Ordinal Categorical', 'Nominal Categorical']
        feature_type = pd.Series(feature_type)

        feature_type_count = [len(target_col), len(cat_cols), len(num_cols), len(bin_cols), len(multi_cols), len(ord_cols), len(nom_cols)]
        feature_type_count = pd.Series(feature_type_count)

        # display(Markdown(f"### <u>Overview of Feature Types</u>"))
        feature_type_df = pd.concat([feature_type, feature_type_count], axis = 1, keys = ['Features Type', 'Frequency'])
        feature_type_df.sort_values(by='Frequency', ascending=False, inplace=True)
        feature_type_df.set_index(keys='Features Type', inplace=True)
        return feature_type_df, early_fn, not_fn_early


    # In[53]:


    feature_type_df, early, not_early = update_category_numeric_cols()
    feature_type_df


    # In[54]:


    # # https://stackoverflow.com/questions/53717543/reduce-number-of-plots-in-sns-pairplot

    # from IPython.display import Markdown
    # from scipy.stats import pearsonr

    # hue = 'EarlyClaimAndLapsed'
    # vars_per_line = 4
    # all_vars = num_cols #list(claim_df.columns.symmetric_difference([hue]))
    # # all_vars.remove('DiagCode')

    # for var in all_vars:
    #     rest_vars = list(all_vars)
    #     rest_vars.remove(var)
    #     display(Markdown(f"## {var}"))
    #     corr_lst = []
    #     while rest_vars:
    #         line_vars = rest_vars[:vars_per_line]
    #         del rest_vars[:vars_per_line]
    #         line_var_names = ", ".join(line_vars)
    #         display(Markdown(f"### {var} vs {line_var_names}"))
    #         sns.pairplot(claim_df, x_vars=line_vars, y_vars=[var], hue=hue, palette='bright', height=3)
    #         plt.show()
    #         plt.close()
    #         for x in line_vars:
    #             corr, _ = pearsonr(claim_df[x], claim_df[var], )
    #             corr_lst.append(round(corr, 3))
    #         if len(corr_lst) == 4:
    #             a, b, c, d = corr_lst
    #             display(Markdown(f"**Correlation: &emsp; &ensp; &ensp; &ensp;{a} &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &ensp;{b} &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &ensp;{c} &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; &ensp;{d}**"))
    #         elif len(corr_lst) == 3:
    #             a, b, c = corr_lst
    #             display(Markdown(f"**Correlation: &emsp; &ensp; &ensp; &ensp;{a} &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &ensp;{b} &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &ensp;{c}**"))
    #         elif len(corr_lst) == 2:
    #             a, b = corr_lst
    #             display(Markdown(f"**Correlation: &emsp; &ensp; &ensp; &ensp;{a} &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; &ensp;{b}**"))
    #         else:
    #             a = corr_lst[0]
    #             display(Markdown(f"**Correlation: &emsp; &ensp; &ensp; &ensp;{a}**"))
    #         corr_lst = []


    # # 3 Data Preparation
    # ## 3.1 Selecting Data
    # 
    # **Selecting items (rows):** In our dataset, there are some outliers which seriously affect the accuracy of our prediction. So, we will remove those outliers in our dataset in order to gain a more accurate and precise prediction when performing modelling. Besides, we will also check whether there are any missing values, and we will take correspondence measures either to remove or replace it with mean values (if any). Please refer to **3.2 Data Cleaning** for further elaboration and explanation.
    # 
    # **Selecting attributes (columns):** In our data set, there are 52 features (columns), and the last feature called "EarlyClaimAndLapsed" is selected as the outcome. The rest 51 features in our dataset are selected as the independent variable to predict the outcome. During the data preparation stage, the categorical and continuous data have been identified and both types of data will undergo constant features checking, but an extra checking called highly concentrated features is applied on continuous data. The continuous data will then undergo the data normalisation or standardisation process so that the final value is between 0 and 1.

    # ## 3.2 Data Cleaning
    # We usually can't have any missing values in our dataset if we want to use them for predictive modeling. This is because when performing predictive modeling, it may affect accuracy and precision, thereby reducing the reliability and trustworthiness towards the result. Therefore, we will check whether there are any missing values in our dataset first. In our case, there are no missing values in our dataset as we have already remove it before this **(Data Manipulation under 2.2.6)**. Besides, we will also check for the occurrence of outliers and remove the outliers using Inter-Quartile Range (IQR).

    # ### 3.2.1 Remove Missing Values

    # In[55]:


    # Check missing values
    # display(Markdown(f"**Total missing values: {claim_df.isnull().sum().sum()}**"))
    # display(Markdown(f"We don't have any missing values in our dataset because we have removed them at an early stage."))


    # ### 3.2.2 Remove Constant & Low Variance Features
    # **Constant Features:** Means that the feature has only 1 single value.<br>
    # **Low Variance Features:** Means the mode value representing the feature occupies a large proportion of the dataset.

    # In[56]:


    # display(Markdown(f"### Total Rows: {claim_df.shape[0]}"))

    mode_df = claim_df[num_cols].mode().iloc[:1, :].T
    mode_df.reset_index(inplace=True)
    mode_df.rename(columns = {'index':'Continuous Features', 0:'Mode Value'}, inplace=True)

    concen_df = pd.DataFrame(claim_df[num_cols].var(), columns=['Variance']).reset_index().rename(columns = {'index':'Continuous Features'})
    concen_df.drop('Variance', axis=1, inplace=True)
    concen_df = pd.merge(concen_df, mode_df, on='Continuous Features')
    concen_df['Frequency'] = concen_df['Continuous Features'].apply(lambda x: claim_df[x].value_counts().tolist()[0])
    concen_df.sort_values(by=['Frequency'], inplace=True, ascending=False)
    concen_df['Percentage (%)'] = concen_df['Frequency'].apply(lambda x: round(x / claim_df.shape[0] * 100, 2))
    concen_df.reset_index(drop=True, inplace=True)
    concen_df


    # From the above table, we found that there are **no numeric constant features** in our dataset, but we found that there are many numeric features whose mode has a very high number of occurences where these features will not give any info to the model while training the model. To combat this issue, we decided to **remove those numeric features whose mode occurrences occupy 76% or more** in the dataset.

    # In[57]:


    # Find those continuous features whose mode occurrences occupy 76% or more in the dataset
    concen_df = concen_df.loc[concen_df["Percentage (%)"] >= 76, ['Continuous Features']]
    concen_df


    # In[58]:


    # Drop above features in our dataset
    if len(concen_df.iloc[:, 0].tolist()) > 0:
        claim_df.drop(labels=concen_df.iloc[:, 0].tolist(), axis=1, inplace=True)
        print(f"Successfully drop {concen_df.iloc[:, 0].tolist()}")
    else:
        print(f"No low variance numeric features (a.k.a. columns) are needed to drop.")


    # Above steps are to remove **constant & low variance numerical features**. Now, we are **going to do the same thing for categorical fatures**. However, for categorical features, we just need to check whether there are any **contant features only** which mean **contains only 1 category for a categorical feature**. 

    # In[59]:


    # display(Markdown(f"### Total Rows: {claim_df.shape[0]}"))

    mode_df = claim_df[cat_cols].mode().iloc[:1, :].T
    mode_df.reset_index(inplace=True)
    mode_df.rename(columns = {'index':'Categorical Features', 0:'Mode Value'}, inplace=True)

    # concen_df = pd.DataFrame(claim_df[num_cols].var(), columns=['Variance']).reset_index().rename(columns = {'index':'Continuous Features'})
    # concen_df.drop('Variance', axis=1, inplace=True)
    # concen_df = pd.merge(concen_df, mode_df, on='Continuous Features')

    mode_df['Frequency'] = mode_df['Categorical Features'].apply(lambda x: claim_df[x].value_counts().tolist()[0])
    mode_df.sort_values(by=['Frequency'], inplace=True, ascending=False)
    mode_df['Percentage (%)'] = mode_df['Frequency'].apply(lambda x: round(x / claim_df.shape[0] * 100, 2))
    mode_df.reset_index(drop=True, inplace=True)
    mode_df


    # From the above table, we found that there are **8 categorical constant features** in our dataset. So, we are going to remove those categorical constant features.
    # 
    # By right, we also could **remove those categorical features whose mode has a very high number of occurences in the dataset (Low Variance)** especially those **categorical features whose mode occurrences occupy 95% or more in the dataset** as they will not give much info to the model while training the model. 

    # In[60]:


    # Find those category features has only 1 category
    mode_df = mode_df.loc[mode_df["Percentage (%)"] >= 95, ['Categorical Features']]

    # Drop those above columns
    if len(mode_df.iloc[:, 0].tolist()) > 0:
        claim_df.drop(labels=mode_df.iloc[:, 0].tolist(), axis=1, inplace=True)
        print(f"Successfully drop {mode_df.iloc[:, 0].tolist()}")
    else:
        print(f"No low variance categorical features (a.k.a. columns) are needed to drop.")


    # In[61]:


    # # Update numeric columns as we have dropped some numeric columns just now
    # num_cols = claim_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    feature_type_df, early, not_early = update_category_numeric_cols()
    feature_type_df


    # ### 3.2.3 Remove Outliers

    # In[62]:


    # Create Box Plot Function
    def vis_box_plot(claim_df):
        dis = 0
        displayable = True
        count_display = 0
        outliers = None
        for index, ele in enumerate(num_cols):
            if (index + 1) % 10 == 0:
                outliers = claim_df[num_cols[int(f'{dis}'):index + 1]].copy()
                dis = index
            
            elif len(num_cols) < 10 and ele == num_cols[-1]:
                outliers = claim_df[num_cols[int(f'{dis}'):]].copy()
                
            elif ele == num_cols[-1]:
                outliers = claim_df[num_cols[int(f'{dis+1}'):]].copy()
                
            else:
                displayable = False

            if displayable:
                fig = px.box(outliers, height=800, width=1000, points="outliers")
                fig.update_layout(title="<b>Outliers Visualisation for All Continuous Features</b>", 
                                yaxis_title='<b>Values</b>', xaxis_title='<b>Continuous Features</b>', 
                                width=950, height=800, title_x = 0.5, font_size=12)
                fig.show()
            displayable = True

    vis_box_plot(claim_df)


    # In[63]:


    # display(Markdown(f"**Extremely High Data Points are:**"))

    # # Create a function to return a list of extreme outliers
    # def extreme_outliers_lst(numbers, threshold, col_name):
    #     new_list = []
    #     for counter, num in enumerate(numbers[:-1]):
    #         if (numbers[counter+1] - num) >= threshold:
    #             new_list = numbers[counter+1:]
    #             display(Markdown(f" - {col_name} List: {new_list}"))
    #             break
    #     return new_list

    # # Create an empty dictionary to store all the boxplot values (key: feature name, val: list of boxplot values for an feature)
    # my_dict = {}

    # # Continuous features list
    # columns_lst = claim_df[num_cols].columns.tolist()

    # # Store the result into the "my_dict" dictionary
    # for col_name, bp_values_lst in zip(columns_lst, upper_outliers):
    #     bp_values_lst.sort()
    #     my_dict[col_name] = extreme_outliers_lst(bp_values_lst, 18008.00, col_name)

    # # Remove extremely high data points
    # for col_name, lst_of_value_to_remove in my_dict.items():
    #     if len(lst_of_value_to_remove) > 0:
    #         claim_df = claim_df.loc[(claim_df[col_name] < lst_of_value_to_remove[0])]

    # # Visualise the boxplot again to see the result
    # vis_box_plot(claim_df)


    # <u>**Remove outliers using Interquartile Range (IQR)**</u><br>
    # IQR is the acronym for Interquartile Range. It **measures the statistical dispersion of the data values as a measure of overall distribution**. IQR is equivalent to the difference between the first quartile (Q1) and the third quartile (Q3) respectively.
    # 
    # Here, Q1 refers to the first quartile i.e. 25% and Q3 refers to the third quartile i.e. 75%. We will be using Boxplots to detect and visualize the outliers present in the dataset. Boxplots depict the distribution of the data in terms of quartiles and consists of the following components:
    # 
    #  - Q1: 25%
    #  - Q2: 50%
    #  - Q3: 75%
    #  - Lower bound/whisker
    #  - Upper whisker/bound
    # 
    # ![image-2.png](attachment:image-2.png)
    # 
    # **Source:** [Detection and Removal of Outliers in Python – An Easy to Understand Guide](https://www.askpython.com/python/examples/detection-removal-outliers-in-python)

    # In[64]:


    # Create outliers removal function using Interquartile Range (IQR)
    def remove_outliers(claim_df, column_name=None):
                
        if column_name is not None:
            Q1 = claim_df[column_name].quantile(0.25)
            Q3 = claim_df[column_name].quantile(0.75)
            if Q3 != 0 and str(Q3) != 'nan' and Q3 != Q1: 
                IQR = Q3 - Q1
                lower_whisker = Q1 - (1.5 * IQR)
                upper_whisker = Q3 + (1.5 * IQR)
                # display(Markdown(f"<u>**{column_name}**</u>"))
                print(f'Q1           : {Q1}\nQ3           : {Q3}\nIQR          : {IQR}\nlower_whisker: {lower_whisker}\nupper_whisker: {upper_whisker}\n')
                claim_df = claim_df.loc[(claim_df[column_name] > lower_whisker) & (claim_df[column_name] < upper_whisker)]
        else:
            for col_name, col_type in zip(claim_df.columns.tolist(), claim_df.dtypes.tolist()):
                if (col_type == 'int64' or col_type == 'float64') and column_name is None:
                    Q1 = claim_df[col_name].quantile(0.25)
                    Q3 = claim_df[col_name].quantile(0.75)
                    if Q3 != 0 and str(Q3) != 'nan' and Q3 != Q1:
                        IQR = Q3 - Q1
                        lower_whisker = Q1 - (1.5 * IQR)
                        upper_whisker = Q3 + (1.5 * IQR)
                        # display(Markdown(f"<u>**{col_name}**</u>"))
                        print(f'Q1           : {Q1}\nQ3           : {Q3}\nIQR          : {IQR}\nlower_whisker: {lower_whisker}\nupper_whisker: {upper_whisker}\n')
                        claim_df = claim_df.loc[(claim_df[col_name] > lower_whisker) & (claim_df[col_name] < upper_whisker)]
        return claim_df

    # # Remove outliers using Interquartile Range (IQR)
    # claim_df = remove_outliers(claim_df)

    # # Visualise the result
    # vis_box_plot(claim_df)


    # From the boxplot above, we can see that **most of the outliers have been successfully removed**. However, we also could see that there are **some features still contaning quite a number outliers**. Therefore, we decided to **remove more outliers** from the features that still containing quite a lot of outliers.
    # 
    # **Source:** [How To Fetch The Exact Values From A Boxplot (Python)](https://towardsdatascience.com/how-to-fetch-the-exact-values-from-a-boxplot-python-8b8a648fc813)

    # In[65]:


    # Create outliers remaning checking function
    def check_remaining_outliers(claim_df, num_cols):
        
        ## Create a boxplot using matplotlib
        bp = plt.boxplot(claim_df[num_cols], showmeans=True) # ['SASurgeonExpPay', 'HSSExpPayTotal', 'HABMedExpPay']

        ## Retrieve Q1 value for each feature
        q1 = [round(min(item.get_ydata()), 1) for item in bp['boxes']]

        ## Retrieve outliers for each feature
        fliers = [item.get_ydata() for item in bp['fliers']]
        lower_outliers = []
        upper_outliers = []
        for i in range(len(fliers)):
            lower_outliers_by_box = []
            upper_outliers_by_box = []
            for outlier in fliers[i]:
                if outlier < q1[i]:
                    lower_outliers_by_box.append(round(outlier, 4))
                else:
                    upper_outliers_by_box.append(round(outlier, 4))
            
            lower_outliers_by_box.sort(reverse=True)
            lower_outliers.append(lower_outliers_by_box)
            
            upper_outliers_by_box.sort()
            upper_outliers.append(upper_outliers_by_box)
            # print(f'Lower outliers: {lower_outliers}\n'f'Upper outliers: {upper_outliers}')
        return lower_outliers, upper_outliers

    lower_outliers, upper_outliers = check_remaining_outliers(claim_df, num_cols)


    # In[66]:


    def chkList(lst):
        if len(lst) >= 8:
            return True
        else:
            return False

    def chkListLength(lst):
        for index, inner_lst in enumerate(lst):
            if len(inner_lst) >= 15: # more than 15 outliers data points
                return True
        return False
        
    upper_outliers_col_name = claim_df.select_dtypes(include=['number']).columns.tolist()

    count = 0
    while count < 4: # chkListLength(upper_outliers) == True:
        
        count2 = 0
        
        while count2 < len(upper_outliers_col_name):
            
            if chkList(upper_outliers[count2]) == True:
                # Remove outliers
                claim_df = remove_outliers(claim_df, upper_outliers_col_name[count2])
            count2 += 1
        
        # Check is there any outliers left
        lower_outliers, upper_outliers = check_remaining_outliers(claim_df, num_cols)
        
        # display(Markdown(f"### **<u>Loop {count+1} in Removing Outliers</u>**"))
        count += 1
        
        # Visualise the result
        vis_box_plot(claim_df)
        
        print()


    # In[67]:


    lower_outliers[1]


    # In[68]:


    # update category and numeric columns as we have dropped some columns just now
    feature_type_df, early, not_early = update_category_numeric_cols()
    feature_type_df


    # ### 3.2.4 Handling High Cardinality Features
    # Only **applicable to categorical features**
    # 
    # <u><b>Source:</b></u>
    # 1. [Dealing with features that have high cardinality](https://towardsdatascience.com/dealing-with-features-that-have-high-cardinality-1c9212d7ff1b)

    # In[69]:


    # Reset the index of dataframe so that the index values can be moved out as the first column in the dataframe.
    # display(Markdown(f"#### Unique values of each categorical features:"))
    cat_unique_df = pd.DataFrame(claim_df[claim_df.select_dtypes(include = ['object']).keys().tolist()].nunique(), 
                                columns=['Number of Unique Values']).reset_index().rename(columns = {'index':'Categorical Features'}).sort_values(by=['Number of Unique Values'], ascending = False)
    cat_unique_df.reset_index(inplace=True, drop=True)
    cat_unique_df


    # In[70]:


    # Select those more than 6 unique categories
    cat_unique_df.loc[cat_unique_df['Number of Unique Values'] > 6]


    # In[71]:


    lst = cat_unique_df.loc[cat_unique_df['Number of Unique Values'] > 6]['Categorical Features'].tolist() + ['EarlyClaimAndLapsed']

    # sweet_report = sv.analyze(source=claim_df[lst], target_feat='EarlyClaimAndLapsed')
    # sweet_report.show_notebook(w="100%", h="Full", layout='vertical')


    # In[72]:


    # Drop 'SELL_AGENT_OCCUPATION_CODE' and 'SERVICE_AGENT_OCCUPATION_CODE' columns as 
    # these columns have too many unique category values 

    if len(cat_unique_df.loc[cat_unique_df['Number of Unique Values'] > 18]['Categorical Features'].tolist()) != 0:
        claim_df.drop(labels=cat_unique_df.loc[cat_unique_df['Number of Unique Values'] > 18]['Categorical Features'].tolist(), 
                    axis=1, inplace=True) # 'BRANCH_CODE'
        print(f"Successfully drop {cat_unique_df.loc[cat_unique_df['Number of Unique Values'] > 18]['Categorical Features'].tolist()}")
    else:
        print(f"No high cardinality (>18) categorical features (a.k.a. columns) are needed to drop.")


    # In[73]:


    # con_cat_df = pd.concat({'Frequencies': claim_df['RISK_CODE'].value_counts(ascending=True), 
    #                         'Percentage': claim_df['RISK_CODE'].value_counts(normalize=True, ascending=True)}, 
    #                        axis=1
    #                       ).reset_index().rename(columns = {'index':'Value'})

    # # First unique value that its percentage is >= 1%
    # first_serie_cat = con_cat_df.loc[con_cat_df['Percentage'] >= 0.01].iloc[0, :]['Frequencies']
    # sum = 0
    # col_other = []
    # for counter, (val, freq, percent) in enumerate(zip(con_cat_df['Value'].tolist(), con_cat_df['Frequencies'].tolist(), con_cat_df['Percentage'].tolist())):
    #     if percent < 0.01:
    #         sum += freq
    #         col_other.append(val)
    #         if sum > first_serie_cat:
    #             sum -= freq
    #             col_other.pop()
    #             break
    # print(col_other)


    # In[75]:


    # Shows the result of 'RISK_CODE' column before reducing the unique features
    # display(Markdown(f"**Before reducing the unique features for `RISK_CODE` column:**"))
    test_df = pd.concat({'Frequencies': claim_df['RISK_CODE'].value_counts(ascending=True), 
                            'Percentage': claim_df['RISK_CODE'].value_counts(normalize=True, ascending=True)}, 
                        axis=1
                        ).reset_index().rename(columns = {'index':'Value'})
    test_df #.iloc[3:, :].Percentage.sum()


    # In[76]:


    from collections import Counter
    def cumulatively_categorise(column, threshold=0.75, return_categories_list=True):
        
        # Find the threshold value using the percentage and number of instances in the column
        threshold_value=int(threshold * len(column))
        
        # Initialise an empty list for our new minimised categories
        categories_list = []
        
        # Initialise a variable to calculate the sum of frequencies
        s = 0
        
        # Create a counter dictionary of the form unique_value: frequency
        counts = Counter(column)

        #Loop through the category name and its corresponding frequency after sorting the categories by descending order of frequency
        for i, j in counts.most_common():
            # Add the frequency to the global sum
            s += dict(counts)[i]
            # Append the category name to the list
            categories_list.append(i)
            # Check if the global sum has reached the threshold value, if so break the loop
            if s >= threshold_value:
                break
        
        # Append the category Other to the list
        categories_list.append('Other')

        # Replace all instances not in our new categories by Other  
        new_column = column.apply(lambda x: x if x in categories_list else 'Other')

        # Return transformed column and unique values if return_categories=True
        if(return_categories_list):
            return new_column, categories_list
            # Return only the transformed column if return_categories=False
        else:
            return new_column


    # In[78]:


    # high_card_cols = {'STATE': 0.9898413581964931, 
    #                   'SELL_AGENT_STATE': 0.9913999443362094, 
    #                   'RISK_CODE': 0.9857500695797383, 
    #                   'SERVICE_AGENT_STATE': 0.9905649874756469
    #                  }

    high_card_cols = {'STATE': 0.9872200082451559, 
                    'SELL_AGENT_STATE': 0.9894187165040538, 
                    'RISK_CODE': 0.9891438779716917, 
                    'SERVICE_AGENT_STATE': 0.9890064587055105
                    }

    for col, thresh in high_card_cols.items():
        transformed_column, new_category_list = cumulatively_categorise(column=claim_df[col], threshold=thresh, return_categories_list=True)
        claim_df[col] = transformed_column


    # In[79]:


    # Shows the result of 'RISK_CODE' column before reducing the unique features
    # display(Markdown(f"**After reducing the unique features for `RISK_CODE` column:**"))
    pd.concat({'Frequencies': claim_df['RISK_CODE'].value_counts(ascending=True), 
                            'Percentage': claim_df['RISK_CODE'].value_counts(normalize=True, ascending=True)}, 
                        axis=1
                        ).reset_index().rename(columns = {'index':'Value'})


    # ### 3.2.5 Remove Highly Correlated Features

    # In this section, we are going to **remove those highly correlated independent features**. In general, it is recommended to avoid having correlated features in your dataset. This is because a group of highly correlated features **will not bring additional information (or just very few)**, but **will increase the complexity of the algorithm**, thus **increasing the risk of errors**. Depending on the features and the model, correlated features might not always harm the performance of the model but that is a real risk.

    # Codes below are based on this tutorial:
    # 1. [Plotly Heatmap (visualize the correlation matrix as a heatmap)](https://en.ai-research-collection.com/plotly-heatmap/)

    # In[80]:


    # claim_df_corr = claim_df.corr()
    # fig = ff.create_annotated_heatmap(
    #     z = np.array(claim_df_corr),
    #     x = list(claim_df_corr.columns),
    #     y = list(claim_df_corr.index),
    #     annotation_text = np.around(np.array(claim_df_corr), decimals=2),
    #     hoverinfo='z + x + y',
    #     colorscale='Viridis'
    # )
    # fig.show()

    # cat_target_cols = cat_cols + target_col
    # corr_df = pd.DataFrame(associations(dataset=claim_df, numerical_columns=num_cols, nom_nom_assoc='theil', figsize=(20, 20), nominal_columns=cat_target_cols).get('corr'))
    # corr_df


    # In[81]:


    # display(Markdown(f"### <u>Correlation of each features against EarlyClaimAndLapsed:</u>"))
    target_corr_dict = dict()
    target_corr_lst = list()

    # EarlyClaimAndLapsed_idx = corr_df.index.tolist().index('EarlyClaimAndLapsed')

    # for index, col_name in enumerate(corr_df.columns):
        # if col_name in claim_df.columns.tolist()[:-1]:
            # corr_val = corr_df.loc[corr_df.index.tolist()[EarlyClaimAndLapsed_idx], col_name] # getting the correltion value
        #     target_corr_dict[col_name] = corr_val
        #     target_corr_lst.append([col_name, corr_val])

    target_corr_df = pd.DataFrame(target_corr_lst, columns=['Features', 'Correlation Value against EarlyClaimAndLapsed'])
    target_corr_df.sort_values(by=['Correlation Value against EarlyClaimAndLapsed'], ascending=True, inplace=True)
    target_corr_df.reset_index(drop=True, inplace=True)
    target_corr_df


    # In[82]:


    # # with the following function we can select highly correlated features
    # # it will remove the first feature that is correlated with anything other feature

    # def correlation(corr_matrix, threshold):
    #     col_corr = list()  # Set of all the names of correlated columns
    #     for i in range(len(corr_matrix.columns)):
    #         for j in range(i):
    #             if abs(corr_matrix.iloc[i, j]) >= threshold: # we are interested in absolute coeff value
    #                 colname1 = corr_matrix.columns[i]  # getting the name of column
    #                 colname2 = corr_matrix.columns[j]
    #                 col_corr.append([colname1, colname2, corr_matrix.iloc[i, j]])
    #     return col_corr

    # display(Markdown(f"### <u>List of features who correlated 80% and above with each other:</u>"))
    # corr_features = correlation(corr_df, 0.8)
    # corr_df2 = pd.DataFrame(corr_features, columns=['First Feature (FF)', 'Second Feature (SF)', 'Correlation (FF x SF)'])
    # corr_df2.sort_values(by=["Correlation (FF x SF)"], inplace=True)
    # corr_df2.reset_index(drop=True, inplace=True)
    # corr_df2


    # In[83]:


    # corr_df = claim_df_corr.copy()
    # display(Markdown(f"### <u>List of features who correlated 80% and above with each other:</u>"))

    row_index = 0
    corrDict = {}
    row_name = []
    col_name = []
    corr_val = []

    # while row_index < len(corr_df.index.tolist()):
    #     for index, x in enumerate(corr_df.iloc[row_index, :]):
    #         if abs(x) >= 0.8 and index != row_index:
    #             if abs(x) in corr_val:
    #                 if (corr_df.index.tolist()[row_index] in col_name) and (corr_df.columns.tolist()[index] in row_name):
    #                     continue
    #             row_name.append(corr_df.index.tolist()[row_index])
    #             col_name.append(corr_df.columns.tolist()[index])
    #             corr_val.append(x)
    #     row_index += 1
        
    corrDict ={"First Feature (FF)": row_name, "Second Feature (SF)": col_name, "Correlation (FF x SF)": corr_val}
    corr_df2=pd.DataFrame(corrDict)
    corr_df2.drop_duplicates(subset=['First Feature (FF)', 'Second Feature (SF)'], inplace=True, ignore_index=True)
    # corr_df2

    rows_to_be_dropped = []
    looped_lst = []

    # Swab first and second column
    corr_df3 = corr_df2.iloc[:, [1, 0] + list(range(2, corr_df2.shape[1]))]
    corr_df3
    for i in range(len(corr_df3)):
        first_two_col_row_data = [corr_df3.iloc[i, 0], corr_df3.iloc[i, 1]]
        for index in corr_df2.index:
            if first_two_col_row_data == corr_df2.iloc[index, 0:2].tolist():
                val1, val2 = first_two_col_row_data
                looped_lst.append([val2, val1])
                if corr_df2.iloc[index, 0:2].tolist() not in looped_lst:
                    rows_to_be_dropped.append(index)

    corr_df2.drop(index=rows_to_be_dropped, inplace=True)
    corr_df2.reset_index(drop=True, inplace=True)
    corr_df2


    # In[84]:


    # Retrieve all the unique features from above list
    features_set = set()
    for col_name in corr_df2.columns:
        if type(corr_df2[col_name].tolist()[0]) == str:
            for col in corr_df2[col_name].tolist():
                features_set.add(col)
    features_set = list(features_set)
    features_set


    # In[85]:


    target_corr_dict = dict()
    target_corr_lst = list()
# 
    EarlyClaimAndLapsed_idx = corr_df.index.tolist().index('EarlyClaimAndLapsed')

    # for index, col_name in enumerate(corr_df.columns):
    #     if col_name in features_set:
    #         corr_val = corr_df.loc[corr_df.index.tolist()[EarlyClaimAndLapsed_idx], col_name] # getting the correltion value
    #         target_corr_dict[col_name] = corr_val
    #         target_corr_lst.append([col_name, corr_val])

    target_corr_df = pd.DataFrame(target_corr_lst, columns=['Features', 'Correlation Value against EarlyClaimAndLapsed'])
    target_corr_df.sort_values(by=['Correlation Value against EarlyClaimAndLapsed'], ascending=True, inplace=True)
    target_corr_df.reset_index(drop=True, inplace=True)
    target_corr_df


    # So far until now, we have found out there are some features that are highly correlated with each other. So, we can actually just pick one of them to drop when we know that the 2 features are highly correlated. However, this may lead us to pick the one with higher correlation with our dependent feature, and we want to avoid this from happening because **the higher the correlation between an independent feature and a dependent feature, the greater the importance of the independent feature in predicting the dependent feature value**.
    # 
    # Therefore, instead of just randomly picking one feature to drop, we can compare those 2 highly correlated features to see **which one has a lower correlation with the dependent feature and then drop it**. Table below shows overall correlation analysis of features that correlate **80% and above** with other and all the yellow highlighted frames are features that are going to be dropped as they have lower correlation with dependent feature comparing to other independent features that they highly correlated with.

    # In[86]:


    # display(Markdown(f"### <u>Correlation analysis of features that correlate 80% and above with each other:</u>"))

    corr_df2['Correlation (FF x EarlyClaimAndLapsed)'] = corr_df2['First Feature (FF)'].map(target_corr_dict)
    corr_df2['Correlation (SF x EarlyClaimAndLapsed)'] = corr_df2['Second Feature (SF)'].map(target_corr_dict)
    FFxSFcorr = corr_df2['Correlation (FF x SF)']
    corr_df2.drop('Correlation (FF x SF)', axis=1, inplace=True)
    corr_df2.insert(loc=corr_df2.shape[1], column='Correlation (FF x SF)', value=FFxSFcorr)
    corr_df2.style.highlight_min(subset=['Correlation (FF x EarlyClaimAndLapsed)', 'Correlation (SF x EarlyClaimAndLapsed)'], axis=1)


    # In[87]:


    col_set_to_drop = set()
    for i in range(len(corr_df2)):
        min_num = min(corr_df2.iloc[i, 2].item(), corr_df2.iloc[i, 3].item())
        
        if str(corr_df2.iloc[i, 4].item())[0] == '-':
            continue
        elif min_num == corr_df2.iloc[i, 2]:
            col_set_to_drop.add(corr_df2.iloc[i, 0])
        else:
            col_set_to_drop.add(corr_df2.iloc[i, 1])
            
    # display(Markdown(f"**Columns to be dropped:**"))
    col_set_to_drop


    # In[88]:


    if 'ENTRY_AGE' in col_set_to_drop:
        col_set_to_drop.remove('ENTRY_AGE')
    col_set_to_drop


    # In[89]:


    # Drop all the selected columns above
    claim_df.drop(labels=col_set_to_drop, inplace=True, axis=1)
    claim_df


    # In[90]:


    # update category and numeric columns as we have dropped some columns just now
    feature_type_df, early, not_early = update_category_numeric_cols()
    feature_type_df


    # Remove all of the category values from each category variable if the frequency of that particular category value under its category feature **<= 5**.

    # In[91]:


    # multi = [col for col, val in (claim_df.select_dtypes(include = ['object']).nunique() > 2).items() if val is True]
    # multi

    # Remove the unique category value from each category variable if the frequency of the category value 
    # is <= 5 in that variable
    for col in multi_cols:
        for col_name, count in claim_df[col].value_counts().items():
            if count <= 5:
                claim_df = claim_df.loc[(claim_df[col] != col_name)]
                claim_df.reset_index(drop=True, inplace=True)


    # In[92]:


    # update category and numeric columns as we have dropped some columns just now
    feature_type_df, early, not_early = update_category_numeric_cols()
    feature_type_df


    # In[93]:


    # Check duplicate rows again
    if claim_df[claim_df.duplicated()].shape[0] > 0:
        print(f"There are {claim_df[claim_df.duplicated()].shape[0]} duplicated rows.")
        print("\nDuplicate rows will be cleared now...")
        claim_df.drop_duplicates(inplace=True)
    else:
        print("Congratz! No duplicate rows.")


    # #### Overall Data Review After Preprocessing

    # In[94]:


    pie_display = claim_df

    # labels
    lab = pie_display["EarlyClaimAndLapsed"].value_counts().keys().tolist()

    # values
    val = claim_df["EarlyClaimAndLapsed"].value_counts().values.tolist()

    trace = go.Pie(labels = lab , values = val, marker = dict(colors = ['blue' ,'green'], line = dict(color = "white", width =  1.3)), rotation = 90, hoverinfo = "label + value + text", hole = 0.5)
    layout = go.Layout(dict(title = "Distribution of Early & Lapsed Claimed in Data", title_x = 0.5, plot_bgcolor  = "rgb(243, 243, 243)", paper_bgcolor = "rgb(243, 243, 243)",))

    import plotly.io as pio
    pio.renderers.default='notebook'

    fig = go.Figure(data = [trace], layout = layout)
    py.iplot(fig)


    # In[95]:


    # Using sweetwiz to generate report
    # import warnings
    # warnings.filterwarnings('ignore')

    # sweet_report = sv.analyze(source=claim_df, target_feat='EarlyClaimAndLapsed')
    # sweet_report.show_notebook(w="100%", h="Full", layout='vertical')


    # In[96]:


    for cat_col in claim_df.select_dtypes(include = ['object']).keys().tolist():
        
        if cat_col != 'EarlyClaimAndLapsed':
            df_g = claim_df[[cat_col, 'EarlyClaimAndLapsed']].groupby([cat_col, 'EarlyClaimAndLapsed']).size().reset_index()
            df_g['percentage'] = claim_df[[cat_col, 'EarlyClaimAndLapsed']].groupby([cat_col, 'EarlyClaimAndLapsed']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = [cat_col, 'EarlyClaimAndLapsed', 'Counts', 'Percentage']
            df_g = df_g.sort_values(['Counts'], ascending=False).reset_index(drop=True)
            
            fig = px.bar(df_g, x=cat_col, y=['Counts'], color='EarlyClaimAndLapsed', text=df_g['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)), barmode='group')
            fig.update_layout(title_text=f'{cat_col} vs EarlyClaimAndLapsed', title_x = 0.5, font_size=12)
            fig.show()


    # In[97]:


    lapsed_df = claim_df[claim_df['EarlyClaimAndLapsed']=='Y']
    non_lapsed_df = claim_df[claim_df['EarlyClaimAndLapsed']=='N']

    for con_col in claim_df.select_dtypes(include = ['number']).keys().tolist():

        fig = go.Figure()
        fig.add_trace(go.Histogram(x = non_lapsed_df[con_col], name='N', marker_color='blue', hovertext=non_lapsed_df['EarlyClaimAndLapsed']))
        fig.add_trace(go.Histogram(x = lapsed_df[con_col], name='Y', marker_color='red', hovertext=lapsed_df['EarlyClaimAndLapsed']))
        fig.update_layout(title=f'{con_col} vs EarlyClaimAndLapsed', title_x=0.5, barmode="overlay", 
                        legend_title='EarlyClaimAndLapsed', xaxis_title=f'{con_col}', yaxis_title='Count'
                        )
        fig.update_traces(opacity=0.5, nbinsx=10, histfunc='count', hovertemplate="<br>".join([
                f"{con_col}"+": %{x}",
                "Count: %{y}", 
                "EarlyClaimAndLapsed: %{hovertext}"
        ]))
        fig.show()


    # In[100]:


    # # https://stackoverflow.com/questions/53717543/reduce-number-of-plots-in-sns-pairplot

    # hue = 'EarlyClaimAndLapsed'
    # vars_per_line = 3
    # all_vars = list(claim_df.columns.symmetric_difference([hue])) # num_cols
    # # all_vars.remove('DiagCode')

    # for var in all_vars:
    #     rest_vars = list(all_vars)
    #     rest_vars.remove(var)
    #     display(Markdown(f"## {var}"))
    #     # corr_lst = []
    #     while rest_vars:
    #         line_vars = rest_vars[:vars_per_line]
    #         del rest_vars[:vars_per_line]
    #         line_var_names = ", ".join(line_vars)
    #         display(Markdown(f"### {var} vs {line_var_names}"))
    #         sns.pairplot(claim_df, x_vars=line_vars, y_vars=[var], hue=hue, palette='bright', height=3)
    #         plt.show()


    # In[101]:


    X = claim_df.drop(labels='EarlyClaimAndLapsed', axis=1)
    y = claim_df['EarlyClaimAndLapsed']

    # Transform categorical features into the appropriate type that is expected by LightGBM
    for c in X.columns:
        col_type = X[c].dtype
        if col_type.name == 'object' or col_type.name == 'category':
            X[c] = X[c].astype('category')

    y = y.astype('category')


    # #### **Train Test Split (80:20)**

    # In[102]:


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_train = pd.DataFrame(data=y_train)
    y_test = pd.DataFrame(data=y_test)


    # In[103]:


    category_cols = X_train.select_dtypes(include=['category']).columns.to_list()
    category_cols


    # ## 3.3 Data Transformation (Feature Engineering & Feature Scaling / Encoding)

    # ### 3.3.1 Feature Encoding (Categorical Data)

    # #### 3.3.1.1 Using **Category Encoders** Python Package
    # **<u>Source:</u>** 
    # 1. [How and When to Use Ordinal Encoder](https://leochoi146.medium.com/how-and-when-to-use-ordinal-encoder-d8b0ef90c28c)
    # 2. [Category Encoders Documentation - Ordinal](http://contrib.scikit-learn.org/category_encoders/ordinal.html)

    # In[104]:


    # from category_encoders import OrdinalEncoder

    # # Show Target Features
    # display(Markdown(f"**<u>{target_col[0]}</u>**"))
    print(f"{claim_df[target_col[0]].unique()}")


    # In[105]:


    # Binary Encoding for target feature which is "EarlyClaimAndLapsed"

    maplist = [{'col': 'EarlyClaimAndLapsed', 'mapping': {'Y': 1, 'N': 0}}]
    oe_enc = OrdinalEncoder(mapping=maplist, cols=target_col)
    y_train = oe_enc.fit_transform(y_train)
    y_test = oe_enc.transform(y_test)

    y_train


    # ## 3.4 Handle Imbalanced Dataset

    # Since we are going to build a **tree based model called LightGBM**, we are not required to balance the dataset as **the model can deal with imbalanced dataset internally** by passing the parameter `is_unbalance=True`. Therefore, we just **print some train test split data shapes details in the cell below**. 

    # In[106]:


    # Tree-based data
    # display(Markdown(f"### Tree-based data"))
    print('Shape of X: {}'.format(X.shape))
    print('Shape of y: {}'.format(y.shape))

    print("Number transactions X_train dataset: ", X_train.shape)
    print("Number transactions y_train dataset: ", y_train.shape)
    print("Number transactions X_test dataset: ", X_test.shape)
    print("Number transactions y_test dataset: ", y_test.shape)


    # # 4. Modelling

    # ## 4.1 LightGBM
    # 
    # **<u>Source:</u>**
    # 1. [LightGBM Model in Python | Tutorial | Machine Learning](https://youtu.be/SW3akc0ho7M)
    # 2. [Kaggler’s Guide to LightGBM Hyperparameter Tuning with Optuna in 2021](https://towardsdatascience.com/kagglers-guide-to-lightgbm-hyperparameter-tuning-with-optuna-in-2021-ed048d9838b5)
    # 3. [You Are Missing Out on LightGBM. It Crushes XGBoost in Every Aspect](https://towardsdatascience.com/how-to-beat-the-heck-out-of-xgboost-with-lightgbm-comprehensive-tutorial-5eba52195997)
    # 4. [lightgbm_simple.py](https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_simple.py)
    # 5. [lightgbm.LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.fit)
    # 6. [Metric Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters)
    # 7. [lightgbm.train](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html)
    # 8. [TPS-Mar21, Leaderboard %14, XGB, CatBoost, LGBM + Optuna 🚀](https://medium.com/databulls/tps-mar21-leaderboard-14-xgb-catboost-lgbm-optuna-cdffb5124368)
    # 9. [What is the proper usage of scale_pos_weight in xgboost for imbalanced datasets?](https://stats.stackexchange.com/questions/243207/what-is-the-proper-usage-of-scale-pos-weight-in-xgboost-for-imbalanced-datasets)
    # 10. [XGBoost for multiclassification and imbalanced data](https://stackoverflow.com/questions/67868420/xgboost-for-multiclassification-and-imbalanced-data)
    # 11. [Differences between class_weight and scale_pos weight in LightGBM](https://datascience.stackexchange.com/questions/54043/differences-between-class-weight-and-scale-pos-weight-in-lightgbm)

    # In[348]:


    y_train.value_counts()


    # In[349]:


    from sklearn.utils import class_weight
    y_train_class_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train['EarlyClaimAndLapsed'])
    y_train_class_weights


    # In[350]:


    pd.Series(y_train_class_weights).value_counts()


    # In[351]:


    y_test.value_counts()


    # In[352]:


    y_test_class_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_test['EarlyClaimAndLapsed'])
    y_test_class_weights


    # In[353]:


    pd.Series(y_test_class_weights).value_counts()


    # Using **is_unbalance** parameter (for dealing imbalanced data) to compute metrics for **10 folds** cross validation.

    # In[354]:


    lightgbmc_model = LGBMClassifier(boosting_type='gbdt', objective='binary', random_state=42, is_unbalance=True)

    acc_score = cross_val_score(lightgbmc_model, X_train, y_train.squeeze(), scoring='accuracy', cv=10, n_jobs=-1)
    precision_score = cross_val_score(lightgbmc_model, X_train, y_train.squeeze(), scoring='precision', cv=10, n_jobs=-1)
    f1_score = cross_val_score(lightgbmc_model, X_train, y_train.squeeze(), scoring='f1', cv=10, n_jobs=-1)
    recall_score = cross_val_score(lightgbmc_model, X_train, y_train.squeeze(), scoring='recall', cv=10, n_jobs=-1)
    roc_auc_score = cross_val_score(lightgbmc_model, X_train, y_train.squeeze(), scoring='roc_auc', cv=10, n_jobs=-1)

    print(f"Accuracy: {acc_score.mean()}")
    print(f"Precision: {precision_score.mean()}")
    print(f"F1: {f1_score.mean()}")
    print(f"Recall: {recall_score.mean()}")
    print(f"ROC_AUC: {roc_auc_score.mean()}")


    # Using **scale_pos_weight** parameter (for dealing imbalanced data).

    # In[355]:


    lightgbmc_model = LGBMClassifier(boosting_type='gbdt', objective='binary', random_state=42, scale_pos_weight=(y_train.value_counts()[0])/(y_train.value_counts()[1]))

    acc_score = cross_val_score(lightgbmc_model, X_train, y_train.squeeze().to_numpy(), scoring='accuracy', cv=10, n_jobs=-1)
    precision_score = cross_val_score(lightgbmc_model, X_train, y_train.squeeze().to_numpy(), scoring='precision', cv=10, n_jobs=-1)
    f1_score = cross_val_score(lightgbmc_model, X_train, y_train.squeeze().to_numpy(), scoring='f1', cv=10, n_jobs=-1)
    recall_score = cross_val_score(lightgbmc_model, X_train, y_train.squeeze().to_numpy(), scoring='recall', cv=10, n_jobs=-1)
    roc_auc_score = cross_val_score(lightgbmc_model, X_train, y_train.squeeze().to_numpy(), scoring='roc_auc', cv=10, n_jobs=-1)

    print(f"Accuracy: {acc_score.mean()}")
    print(f"Precision: {precision_score.mean()}")
    print(f"F1: {f1_score.mean()}")
    print(f"Recall: {recall_score.mean()}")
    print(f"ROC_AUC: {roc_auc_score.mean()}")


    # In[356]:


    y_train.squeeze().to_numpy()


    # In[153]:


    # Hyperparameter Tuning using optuna
    from optuna.integration import LightGBMPruningCallback

    def objective(trial, X_train, X_test, y_train, y_test):
        
        param_grid = {
            "boosting_type": trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            "num_leaves": trial.suggest_int("num_leaves", 8, 131072, step=1),
            "max_depth": trial.suggest_int("max_depth", 3, 48),
            'n_estimators': trial.suggest_int('n_estimators', 6000, 11000, step=2),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, step=1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0, step=1),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 200, step=1),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 15),
            "subsample": trial.suggest_float("subsample", 0.2, 0.90, step=0.1),
            "subsample_freq": trial.suggest_categorical("subsample_freq", [1]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.90, step=0.1),
            'cat_smooth' : trial.suggest_int('cat_smooth', 10, 100),
            'cat_l2': trial.suggest_int('cat_l2', 1, 20),
            #"pos_subsample": trial.suggest_float("pos_subsample", 0.2, 0.95, step=0.1),
            #"neg_subsample": trial.suggest_float("neg_subsample", 0.2, 0.95, step=0.1),
            'random_state': 42,
            'metric': 'average_precision',
        }
        
        model = lgbm.LGBMClassifier(objective="binary", scale_pos_weight=(y_train.value_counts()[0])/(y_train.value_counts()[1]), first_metric_only = True, **param_grid)
        model.fit(
            X=X_train,
            y=y_train.squeeze().to_numpy(),
            sample_weight=y_train_class_weights,
            eval_set=[(X_test, y_test.squeeze().to_numpy())],
            eval_metric="average_precision",
            eval_sample_weight=[y_test_class_weights],
            early_stopping_rounds=200,
            verbose=-1,
            feature_name=X.columns.tolist(),
            categorical_feature=category_cols,
            callbacks=[LightGBMPruningCallback(trial, "average_precision")], # Add a pruning callback
        )
        y_preds = model.predict(X_test)
        # y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        return average_precision_score(y_test.squeeze().to_numpy(), y_preds) # roc_auc_score(y_test, y_pred_proba)


    # In[154]:


    study = optuna.create_study(direction="maximize", study_name="LGBM Classifier")
    func = lambda trial: objective(trial, X_train, X_test, y_train, y_test)
    study.optimize(func, n_trials=6500, n_jobs=-1)


    # In[155]:


    print(f'Number of finished trials: {len(study.trials)}')
    print()
    print(f'Best trial:\n{study.best_trial.params}')
    print()
    print(f'Best value: {study.best_value}')


    # In[156]:


    parameters = {}
    parameters['random_state'] = 42
    parameters['metric'] = 'average_precision'

    # display(Markdown(f"#### **<u>Best parameters:</u>**"))
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
        parameters[key] = value


    # In[157]:


    # display(Markdown(f"#### **<u>Best parameters:</u>**"))
    for key, val in parameters.items():
        print(f"{key}: {val}")


    # In[158]:


    lgbm_model = lgbm.LGBMClassifier(objective="binary", scale_pos_weight=(y_train.value_counts()[0])/(y_train.value_counts()[1]), **parameters)

    lgbm_model.fit(
        X=X_train, 
        y=y_train, 
        eval_set=[(X_test, y_test)], 
        eval_metric="auc", 
        early_stopping_rounds=200, 
        categorical_feature=category_cols, 
        verbose=-1
    )


    # ## 4.2 LightGBM (Overall Metrics)

    # ### 4.2.1 Accuracy, Precision, Recall, F1 and AUC Scores

    # In[163]:


    y_train_pred = lgbm_model.predict(X_train)
    # y_train_pred = [1 if x >= 0.5 else 0 for x in y_train_pred]

    y_test_pred = lgbm_model.predict(X_test)
    # y_test_pred = [1 if x >= 0.5 else 0 for x in y_test_pred]

    print("Accuracy Train: {:.4f}\nAccuracy Test: {:.4f}".format(accuracy_score(y_train, y_train_pred),
                                                        accuracy_score(y_test, y_test_pred)))
    print()
    print("Precision Train: {:.4f}\nPrecision Test: {:.4f}".format(precision_score(y_train, y_train_pred),
                                                        precision_score(y_test, y_test_pred)))
    print()
    print("Recall Train: {:.4f}\nRecall Test: {:.4f}".format(recall_score(y_train, y_train_pred),
                                                        recall_score(y_test, y_test_pred)))
    print()
    print("F1 Train: {:.4f}\nF1 Test: {:.4f}".format(f1_score(y_train, y_train_pred),
                                                        f1_score(y_test, y_test_pred)))
    print()
    print("AUC Train: {:.4f}\nAUC Test: {:.4f}".format(roc_auc_score(y_train, lgbm_model.predict_proba(X_train)[:,1]),
                                                        roc_auc_score(y_test, lgbm_model.predict_proba(X_test)[:,1])))


    # In[164]:


    train_score = {}
    test_score = {}

    for x in [0.5, 0.6, 0.7, 0.8, 0.9]:
        y_train_pred = lgbm_model.predict(X_train)
        # y_test_pred = lgbm_model.predict(X_test)

        y_train_pred = [1 if y >= x else 0 for y in y_train_pred]
        # y_test_pred = [1 if y >= x else 0 for y in y_test_pred]
        
        # train
        train_score[x] = [accuracy_score(y_train, y_train_pred), precision_score(y_train, y_train_pred), recall_score(y_train, y_train_pred), f1_score(y_train, y_train_pred), roc_auc_score(y_train, lgbm_model.predict_proba(X_train)[:,1])]
        
        # test
        test_score[x] = [accuracy_score(y_test, y_test_pred), precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), f1_score(y_test, y_test_pred), roc_auc_score(y_test, lgbm_model.predict_proba(X_test)[:,1])]

    # display(Markdown(f'### Train Metrics:'))
    train_score = pd.DataFrame(data=train_score, index=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']).T
    train_score['Threshold'] = train_score.index
    first_column = train_score.pop('Threshold')
    train_score.insert(0, 'Threshold', first_column)
    train_score = train_score.reset_index(drop=True)
    train_score


    # In[165]:


    # display(Markdown(f'### Test Metrics:'))
    test_score = pd.DataFrame(data=test_score, index=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']).T
    test_score['Threshold'] = test_score.index

    first_column = test_score.pop('Threshold')
    test_score.insert(0, 'Threshold', first_column)
    test_score = test_score.reset_index(drop=True)
    test_score


    # ### 4.2.2 LightGBM (Confusion Matrix)

    # In[166]:


    lst = []
    for x in [0.5, 0.6, 0.7, 0.8, 0.9]:
        y_test_pred = lgbm_model.predict(X_test)
        # y_test_pred = [1 if y > x else 0 for y in y_test_pred]
        
        cm = confusion_matrix(y_test, y_test_pred)
        
        group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        lst.append((cm, labels))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Confusion Matrix by Threshold', fontsize=15)

    sns.heatmap(ax=axes[0, 0], data=lst[0][0], xticklabels = ['Predicted_No', 'Predicted_Yes'], yticklabels = ['Actual_No', 'Actual_Yes'], annot = lst[0][1], fmt = '', annot_kws = {'fontsize':20}, cmap="YlGnBu", square = True)
    axes[0, 0].set_title("LightGBM Classifier (Thresh: 0.5)")

    sns.heatmap(ax=axes[0, 1], data=lst[1][0], xticklabels = ['Predicted_No', 'Predicted_Yes'], yticklabels = ['Actual_No', 'Actual_Yes'], annot = lst[1][1], fmt = '', annot_kws = {'fontsize':20}, cmap="YlGnBu", square = True)
    axes[0, 1].set_title("LightGBM Classifier (Thresh: 0.6)")

    sns.heatmap(ax=axes[0, 2], data=lst[2][0], xticklabels = ['Predicted_No', 'Predicted_Yes'], yticklabels = ['Actual_No', 'Actual_Yes'], annot = lst[2][1], fmt = '', annot_kws = {'fontsize':20}, cmap="YlGnBu", square = True)
    axes[0, 2].set_title("LightGBM Classifier (Thresh: 0.7)")

    sns.heatmap(ax=axes[1, 0], data=lst[3][0], xticklabels = ['Predicted_No', 'Predicted_Yes'], yticklabels = ['Actual_No', 'Actual_Yes'], annot = lst[3][1], fmt = '', annot_kws = {'fontsize':20}, cmap="YlGnBu", square = True)
    axes[1, 0].set_title("LightGBM Classifier (Thresh: 0.8)")

    sns.heatmap(ax=axes[1, 1], data=lst[4][0], xticklabels = ['Predicted_No', 'Predicted_Yes'], yticklabels = ['Actual_No', 'Actual_Yes'], annot = lst[4][1], fmt = '', annot_kws = {'fontsize':20}, cmap="YlGnBu", square = True)
    axes[1, 1].set_title("LightGBM Classifier (Thresh: 0.9)")


    # In[167]:


    for x in [0.5, 0.6, 0.7, 0.8, 0.9]:
        y_test_pred = lgbm_model.predict(X_test)
        # y_test_pred = [1 if y > x else 0 for y in y_test_pred]
        
        cm = confusion_matrix(y_test, y_test_pred)
        
        group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        
        plt.figure(figsize = (6,6))
        sns.heatmap(cm, xticklabels = ['Predicted_No', 'Predicted_Yes'], yticklabels = ['Actual_No', 'Actual_Yes'],
        annot = labels, fmt = '', annot_kws = {'fontsize':20}, cmap="YlGnBu", square = True)
        plt.title(f"LightGBM Classifier (Thresh: {x})")
        plt.subplots_adjust(wspace = .3,hspace = .3,)


    # ### 4.2.3 LightGBM (ROC AUC Curve)
    # 
    # **<u>Source:</u>**
    # 1. [ROC curve explained](https://towardsdatascience.com/roc-curve-explained-50acab4f7bd8)
    # 2. [How to Calculate AUC (Area Under Curve) in Python](https://www.statology.org/auc-in-python/)
    # 3. [How to Interpret a ROC Curve (With Examples)](https://www.statology.org/interpret-roc-curve/)
    # 4. [How to Plot a ROC Curve in Python (Step-by-Step)](https://www.statology.org/plot-roc-curve-python/)
    # 5. [What is Considered a Good AUC Score?](https://www.statology.org/what-is-a-good-auc-score/)

    # In[168]:


    def Calculate_Plot_ROC(y_test, y_pred, name):
        # Calculate ROC curve
        fpr, tpr, thr = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr,tpr)

        # Plot the ROC curve
        plt.plot([0, 1], [0, 1], 'k--', label = 'Random')
        plt.plot(fpr, tpr, label ='ROC Curve (Area = %0.2f)' %roc_auc)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'ROC Curve for {name}')
        plt.grid()
        plt.legend(loc ='best')


    # In[169]:


    Calculate_Plot_ROC(y_test, lgbm_model.predict_proba(X_test)[:, 1], 'LightGBM Classifier')


    # ### 4.2.4 LightGBM (Feature Importance)

    # In[170]:


    lgbm.plot_importance(booster=lgbm_model, max_num_features=X_train.shape[1])

    df_feature_importance = (
        pd.DataFrame({
            'feature': lgbm_model.feature_name_,
            'importance': lgbm_model.feature_importances_,
        })
        .sort_values('importance', ascending=False)
    )
    df_feature_importance


    # In[171]:


    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)


    # ### 4.2.5 LightGBM (Interpretation)
    # <u>**Source:**</u>
    # 1. [LightGBM model explained by shap](https://www.kaggle.com/code/cast42/lightgbm-model-explained-by-shap/comments)
    # 2. [How to interpret and explain your machine learning models using SHAP values](https://m.mage.ai/how-to-interpret-and-explain-your-machine-learning-models-using-shap-values-471c2635b78e)
    # 3. [SHAP Dependence Plots](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Census%20income%20classification%20with%20LightGBM.html#SHAP-Dependence-Plots)
    # 4. [9.5 Shapley Values](https://christophm.github.io/interpretable-ml-book/shapley.html)

    # In[172]:


    for name in X_train.columns:
        try:
            shap.dependence_plot(ind=name, shap_values=shap_values[1], features=X, feature_names=X_train.columns.tolist(), display_features=X)
        except (TypeError, ValueError):
            continue


    # In[173]:


    # X_display, y_display = shap.datasets.adult(display=True)
    # X_display


    # In[174]:


    # Xdd, ydd = shap.datasets.adult()
    # Xdd


    # ### 4.2.6 LightGBM (Save Model)

    # In[175]:


    # Save model
    import joblib
    joblib.dump(lgbm_model, 'lgbm_model_avg_precision.pkl')


    # ### 4.2.7 LightGBM (Load Model)

    # In[108]:


    # Load model
    import joblib
    lgbm_model = joblib.load('lgbm_model_auc.pkl')


    # In[109]:


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_train = pd.DataFrame(data=y_train)
    y_test = pd.DataFrame(data=y_test)


    # In[110]:


    category_cols = X_train.select_dtypes(include=['category']).columns.to_list()
    category_cols


    # In[111]:


    from category_encoders import OrdinalEncoder

    # Binary Encoding for target feature which is "EarlyClaimAndLapsed"
    maplist = [{'col': 'EarlyClaimAndLapsed', 'mapping': {'Y': 1, 'N': 0}}]
    oe_enc = OrdinalEncoder(mapping=maplist, cols=target_col)
    y_train = oe_enc.fit_transform(y_train)
    y_test = oe_enc.transform(y_test)

    y_train


    # In[112]:


    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, auc
    y_train_pred = lgbm_model.predict(X_train)
    y_test_pred = lgbm_model.predict(X_test)

    print("Accuracy Train: {:.4f}\nAccuracy Test: {:.4f}".format(accuracy_score(y_train, y_train_pred),
                                                        accuracy_score(y_test, y_test_pred)))
    print()
    print("Precision Train: {:.4f}\nPrecision Test: {:.4f}".format(precision_score(y_train, y_train_pred),
                                                        precision_score(y_test, y_test_pred)))
    print()
    print("Recall Train: {:.4f}\nRecall Test: {:.4f}".format(recall_score(y_train, y_train_pred),
                                                        recall_score(y_test, y_test_pred)))
    print()
    print("F1 Train: {:.4f}\nF1 Test: {:.4f}".format(f1_score(y_train, y_train_pred),
                                                        f1_score(y_test, y_test_pred)))
    print()
    print("AUC Train: {:.4f}\nAUC Test: {:.4f}".format(roc_auc_score(y_train, lgbm_model.predict_proba(X_train)[:,1]),
                                                        roc_auc_score(y_test, lgbm_model.predict_proba(X_test)[:,1])))


    # In[113]:


    train_score = {}
    test_score = {}

    for x in [0.5, 0.6, 0.7, 0.8, 0.9]:
        y_train_pred = lgbm_model.predict(X_train)
        # y_test_pred = lgbm_model.predict(X_test)

        y_train_pred = [1 if y >= x else 0 for y in y_train_pred]
        # y_test_pred = [1 if y >= x else 0 for y in y_test_pred]
        
        # train
        train_score[x] = [accuracy_score(y_train, y_train_pred), precision_score(y_train, y_train_pred), recall_score(y_train, y_train_pred), f1_score(y_train, y_train_pred), roc_auc_score(y_train, lgbm_model.predict_proba(X_train)[:,1])]
        
        # test
        test_score[x] = [accuracy_score(y_test, y_test_pred), precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), f1_score(y_test, y_test_pred), roc_auc_score(y_test, lgbm_model.predict_proba(X_test)[:,1])]

    # display(Markdown(f'### Train Metrics:'))
    train_score = pd.DataFrame(data=train_score, index=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']).T
    train_score['Threshold'] = train_score.index
    first_column = train_score.pop('Threshold')
    train_score.insert(0, 'Threshold', first_column)
    train_score = train_score.reset_index(drop=True)
    train_score


    # In[114]:


    # display(Markdown(f'### Test Metrics:'))
    test_score = pd.DataFrame(data=test_score, index=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']).T
    test_score['Threshold'] = test_score.index

    first_column = test_score.pop('Threshold')
    test_score.insert(0, 'Threshold', first_column)
    test_score = test_score.reset_index(drop=True)
    test_score


    # In[115]:


    y_pred = lgbm_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    plt.figure(figsize = (6,6))
    sns.heatmap(cm, xticklabels = ['Predicted_No', 'Predicted_Yes'], yticklabels = ['Actual_No', 'Actual_Yes'],
    annot = labels, fmt = '', annot_kws = {'fontsize':20}, cmap="YlGnBu", square = True)
    plt.title(f"LightGBM Classifier")
    plt.subplots_adjust(wspace = .3,hspace = .3,)


    # In[116]:


    def Calculate_Plot_ROC(y_test, y_pred, name):
        # Calculate ROC curve
        fpr, tpr, thr = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr,tpr)

        # Plot the ROC curve
        plt.plot([0, 1], [0, 1], 'k--', label = 'Random')
        plt.plot(fpr, tpr, label ='ROC Curve (Area = %0.2f)' %roc_auc)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'ROC Curve for {name}')
        plt.grid()
        plt.legend(loc ='best')


    # In[117]:


    Calculate_Plot_ROC(y_test, lgbm_model.predict_proba(X_test)[:, 1], 'LightGBM Classifier')


    # In[118]:


    lgbm.plot_importance(booster=lgbm_model, max_num_features=X_train.shape[1])

    df_feature_importance = (
        pd.DataFrame({
            'feature': lgbm_model.feature_name_,
            'importance': lgbm_model.feature_importances_,
        })
        .sort_values('importance', ascending=False)
    )
    df_feature_importance


    # In[122]:


    lgbm.plot_metric(booster=lgbm_model, metric='auc')


    # In[119]:


    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)


    # In[120]:


    for col_name in X_train.columns:
        try:
            shap.dependence_plot(ind=col_name, shap_values=shap_values[1], features=X, feature_names=X_train.columns.tolist(), display_features=X)
        except (TypeError, ValueError):
            continue



