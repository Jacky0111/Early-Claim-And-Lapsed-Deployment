import numpy as np
import streamlit as st


class DataManipulation:
    # Transform categorical features into the appropriate type that is expected by LightGBM
    @staticmethod
    def objToCat(data):
        for col in data.columns:
            col_type = data[col].dtype
            if col_type.name == 'object' or col_type.name == 'category' or col_type.name == 'bool':
                data[col] = data[col].astype('category')
        return data

    @staticmethod
    def missingValues(data):
        for col in range(data.shape[1]):
            if data.iloc[:, col].dtype == 'object':
                for index, val in enumerate(data.iloc[:, col].values):
                    data.iloc[:, col].values[index] = str(val).strip()
                    if str(val).isspace() or val is None or str(val).lower() == 'nan' or str(val).lower() == 'none':
                        data.iloc[:, col].values[index] = np.NaN
        return data

    @staticmethod
    def extraLoad(data):
        return data.apply(lambda x: False if isinstance(x, float) else (False if x == 0 else True))

    @staticmethod
    def paymentMode(data):
        # 1: Annual, 2: Half-Yearly, 4: Quarterly, 12: Monthly
        data = data.replace({12.: 'Monthly',
                             1.: 'Yearly',
                             2.: 'Half-yearly',
                             4.: 'Quarterly'}).astype(str)
        return data

    @staticmethod
    def riskSumAssured(data):
        return data.astype(int).astype(str)

    @staticmethod
    def agentPostcode(data):
        data = data.astype(float)
        data = DataManipulation.mapPostcode(data)
        return data

    @staticmethod
    def mapPostcode(data):
        i = 0
        drop_list = []

        while i < data.shape[0]:
            if data[i] in range(50000, 60001):
                data[i] = 'KUALA LUMPUR'
            # elif data[i] == '          ':
            #     data[i] = ' '
            elif str(data[i]).startswith('62'):
                data[i] = 'PUTRAJAYA'
            elif str(data[i]).startswith('87'):
                data[i] = 'LABUAN'
            elif data[i] in range(1000, 2801):
                data[i] = 'PERLIS'
            elif data[i] in range(5000, 9811) or data[i] == 14290 or data[i] == 14390 or data[i] == 34950:
                data[i] = 'KEDAH'
            elif data[i] in range(10000, 14401):
                data[i] = 'PENANG'
            elif data[i] in range(15000, 18501):
                data[i] = 'KELANTAN'
            elif data[i] in range(20000, 24301):
                data[i] = 'TERENGGANU'
            elif data[i] in range(25000, 28801) or str(data[i]).startswith('39') or str(data[i]).startswith('49') or \
                    data[i] == 69000:
                data[i] = 'PAHANG'
            elif data[i] in range(30000, 36811):
                data[i] = 'PERAK'
            elif data[i] in range(40000, 48301) or data[i] in range(63000, 63301) or data[i] == 64000 or data[
                i] in range(
                    68000, 68101):
                data[i] = 'SELANGOR'
            elif data[i] in range(70000, 73501):
                data[i] = 'NEGERI SEMBILAN'
            elif data[i] in range(75000, 78301):
                data[i] = 'MELAKA'
            elif data[i] in range(79000, 86901):
                data[i] = 'JOHOR'
            elif data[i] in range(88000, 91301):
                data[i] = 'SABAH'
            elif data[i] in range(93000, 98851):
                data[i] = 'SARAWAK'
            else:
                drop_list.append(data.index[i])

            i += 1
        return data, drop_list
