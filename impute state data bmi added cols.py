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
folder = "Data for states and transitions bmi"
filename = "BMI patient states imputed method LR added cols.xlsx"
fullpath_read = os.path.join(work_dir, folder, filename)
df_init = pd.read_excel(fullpath_read)

print(df_init)
# print(df_init.columns.values.tolist())

### Keep only relevant columns
# Don't use the following columns: ['Patient','Alcohol','Drugs','Race','Allergies','Surgery Type']
# Compute the following columns: ['PreBMI', 'W1BMI', 'W2BMI', 'W3BMI', 'W4BMI', 'W5BMI', 'W6BMI', 'W7BMI']
col_names = ['Gender', 'Age', 'HPV', 'Smoking', 'Chemo_Systemic_Medication', \
    'Primary_Cancer_Type', 'Stage', 'RT Dose', 'Height', \
    'Pre_RT_wt_kg', 'PreRT_pulse', 'PreRT_pain score', 'Pain score_W1', 'Pain score_W2', 'Pain score_W3', \
    'Pain score_W4', 'Pain score_W5', 'Pain score_W6', 'Pain score_W7', 'W1_wt', 'W1_pulse', 'W2_wt', 'W2_pulse', \
    'W3_wt', 'W3_pulse', 'W4_wt', 'W4_pulse', 'W5_wt', 'W5_pulse', 'W6_wt', 'W6_pulse', 'W7_wt', 'W7_pulse', \
    'PreBMI', 'W1BMI', 'W2BMI', 'W3BMI', 'W4BMI', 'W5BMI', 'W6BMI', 'W7BMI', \
    'objective_dermatitis_W1', 'Feeding Tube_W1', 'objective_dermatitis_W2', 'Feeding Tube_W2', \
    'objective_dermatitis_W3', 'Feeding Tube_W3', 'objective_dermatitis_W4', 'Feeding Tube_W4', \
    'objective_dermatitis_W5', 'Feeding Tube_W5', 'objective_dermatitis_W6', 'Feeding Tube_W6', \
    'objective_dermatitis_W7', 'Feeding Tube_W7'
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
                       random_state=0, estimator=estimator, min_value=0, max_value=3)

df_array = imp.fit_transform(df)

df_final = pd.DataFrame(data = df_array, index = range(df_array.shape[0]), columns=df.columns)
# df_final = df_final.round(decimals=0).astype(int)

### If we want to convert data types of specific columns
convert_dict = {}
for week in range(1,8):
    convert_dict[f'objective_dermatitis_W{week}'] = int
    convert_dict[f'Feeding Tube_W{week}'] = int

df_final = df_final.astype(convert_dict)

### Correct data
for week in range(1,8):
    df_final[f'Feeding Tube_W{week}'] = df_final[f'Feeding Tube_W{week}'].replace([2,3,4], 1)

### Save to excel files (using 100 iterations and 'ascending' imputation order)
df_final.to_excel(os.path.join(work_dir, folder, f"BMI patient states imputed method LR added cols imputed method {method}.xlsx"), index=False, float_format="%.4f", sheet_name="MDASI scores")
