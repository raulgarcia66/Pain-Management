import numpy as np
import pandas as pd
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, BayesianRidge
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR

work_dir = os.getcwd()
folder = "Data for states and transitions July"
filename = "pain_data.xlsx"
full_path = os.path.join(work_dir, folder, filename)
df_init = pd.read_excel(full_path)

print(df_init)
# print(df_init.columns.values.tolist())

### Convert categorical data to integers
# print(df_init['Gender'].unique())  # ['Male' 'Female']
# print(df_init['Smoking'].unique())  # ['Former' 'Never' 'Current']
# print(df_init['Chemo_Systemic_Medication'].unique())  # ['Yes' 'None']
# print(df_init['Primary_Cancer_Type'].unique())  # ['oropharynx' 'oralcavity' 'unknownprimary']
# print(df_init['Stage'].unique())  # ['IVA' 'III' 'I' 'II' nan 'IV' 'IVC' 'IVB' 'IIIB' 'II ']

df_init['Gender'] = df_init['Gender'].replace('Male', 1)
df_init['Gender'] = df_init['Gender'].replace('Female', 0)

df_init['Smoking'] = df_init['Smoking'].replace('Never', 0)
df_init['Smoking'] = df_init['Smoking'].replace('Former', 1)
df_init['Smoking'] = df_init['Smoking'].replace('Current', 1)

df_init['Chemo_Systemic_Medication'] = df_init['Chemo_Systemic_Medication'].replace('Yes', 1)
df_init['Chemo_Systemic_Medication'] = df_init['Chemo_Systemic_Medication'].replace('None', 0)

df_init['Primary_Cancer_Type'] = df_init['Primary_Cancer_Type'].replace('oropharynx', 2)
df_init['Primary_Cancer_Type'] = df_init['Primary_Cancer_Type'].replace('oralcavity', 1)
df_init['Primary_Cancer_Type'] = df_init['Primary_Cancer_Type'].replace('unknownprimary', 0)

df_init['Stage'] = df_init['Stage'].replace(['I'], 1)
df_init['Stage'] = df_init['Stage'].replace(['II','II '], 2)
df_init['Stage'] = df_init['Stage'].replace(['III','IIIB'], 3)
df_init['Stage'] = df_init['Stage'].replace(['IV','IVA','IVB', 'IVC'], 4)

### Keep only relevant columns
# Don't use the following columns: ['Patient','Alcohol','Drugs','Race','Allergies','Surgery Type']
# Compute the following columns: ['PreBMI', 'W1BMI', 'W2BMI', 'W3BMI', 'W4BMI', 'W5BMI', 'W6BMI', 'W7BMI']
col_names = ['Gender', 'Age', 'HPV', 'Smoking', 'Chemo_Systemic_Medication', \
    'Primary_Cancer_Type', 'Stage', 'RT Dose', 'Height', \
    'Pre_RT_wt_kg', 'PreRT_pulse', 'PreRT_pain score', 'Pain score_W1', 'Pain score_W2', 'Pain score_W3', \
    'Pain score_W4', 'Pain score_W5', 'Pain score_W6', 'Pain score_W7', 'W1_wt', 'W1_pulse', 'W2_wt', 'W2_pulse', \
    'W3_wt', 'W3_pulse', 'W4_wt', 'W4_pulse', 'W5_wt', 'W5_pulse', 'W6_wt', 'W6_pulse', 'W7_wt', 'W7_pulse', \
    # 'PreBMI', 'W1BMI', 'W2BMI', 'W3BMI', 'W4BMI', 'W5BMI', 'W6BMI', 'W7BMI'
    ]
df = df_init[col_names]
df = df.replace("NA", np.nan)

### Interpolate
estimator = LinearRegression()
method = "LR"

# estimator = BayesianRidge()  # default
# method = "BR"

num_iter = 100
imp = IterativeImputer(missing_values=np.nan, max_iter=num_iter, verbose=2, imputation_order='ascending',
                       random_state=0, estimator=estimator, min_value=0) # max_value=10

df_array = imp.fit_transform(df)

df_final = pd.DataFrame(data = df_array, index = range(df_array.shape[0]), columns=df.columns)
# df_final = df_final.round(decimals=0).astype(int)

### If we want to convert data types of specific columns
convert_dict = {'Gender':int, 'Age':int, 'HPV':int, 'Smoking':int, 'Chemo_Systemic_Medication':int, \
    'Primary_Cancer_Type':int, 'Stage':int, 'RT Dose':int, 'PreRT_pulse':int, 'PreRT_pain score':int}
for week in range(1,8):
    convert_dict[f'Pain score_W{week}'] = int
    convert_dict[f'W{week}_pulse'] = int

df_final = df_final.astype(convert_dict)

### Correct data
# Only 'Stage' from categorical features had nan's
# print(df_final['Stage'].unique())  # [4 3 1 2]
# df_final['Stage'] = df_final['Stage'].replace([5,6,7,8,9,10], 4)

# for week in range(1,8):
#     print(df_final[f'Pain score_W{week}'].unique())
df_final['Pain score_W7'] = df_final['Stage'].replace(11, 4)  # only this needs correcting

### Calculate BMI. Multiply by (100)^2 since our height is in cm
df_final['PreBMI'] = df_final['Pre_RT_wt_kg'] / ( df_final['Height'] **2 ) * 10000
df_final['W1BMI'] = df_final['W1_wt'] / ( df_final['Height'] **2 ) * 10000
df_final['W2BMI'] = df_final['W2_wt'] / ( df_final['Height'] **2 ) * 10000
df_final['W3BMI'] = df_final['W3_wt'] / ( df_final['Height'] **2 ) * 10000
df_final['W4BMI'] = df_final['W4_wt'] / ( df_final['Height'] **2 ) * 10000
df_final['W5BMI'] = df_final['W5_wt'] / ( df_final['Height'] **2 ) * 10000
df_final['W6BMI'] = df_final['W6_wt'] / ( df_final['Height'] **2 ) * 10000
df_final['W7BMI'] = df_final['W7_wt'] / ( df_final['Height'] **2 ) * 10000

### Save to excel files (using 100 iterations and 'ascending' imputation order)
df_final.to_excel(os.path.join(work_dir, folder, f"BMI patient states imputed method {method}.xlsx"), index=False, float_format="%.4f", sheet_name="MDASI scores")
