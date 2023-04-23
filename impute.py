import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

df_init = pd.read_excel("Pain data-new - Raul - Weekly pain imputed.xlsx", sheet_name="Pain data", usecols="A:AL")
col_names = ["Pre_RT_pain_score", "Pain_score_W1","Pain_score_W2","Pain_score_W3","Pain_score_W4","Pain_score_W5","Pain_score_W6","Pain_score_W7",
             "Pre_RT_wt_kg","W1_wt","W2_wt","W3_wt","W4_wt","W5_wt","W6_wt","W7_wt"]
df = df_init[col_names]
# df = df.iloc[:,0:38]
# print(list(df.columns))

df = df.replace("NA", np.nan)
# print(df)
# print(df.iloc[15:20,17])

lr = LinearRegression()
imp = IterativeImputer(missing_values=np.nan, max_iter=100, verbose=2, imputation_order='roman',random_state=0, estimator=lr)  # default estimator is BayesianRidge()
df_array = imp.fit_transform(df)
# print(df_final.size)
# print(df_final)

df_final = pd.DataFrame(data = df_array, index = range(df_array.shape[0]), columns=df.columns)
convert_dict = {"Pre_RT_pain_score": int, "Pain_score_W1": int,"Pain_score_W2": int,"Pain_score_W3": int,"Pain_score_W4": int,"Pain_score_W5": int,"Pain_score_W6": int,"Pain_score_W7": int}
df_final = df_final.astype(convert_dict)
print(df_final)

# raw 2 used LinearRegression() with max_iter 100 (stopped at 94); raw 1 used BayesianRidge() with max_iter 50
# Always make sure file name is updated
df_final.to_excel("Pain data-new - Raul - Python raw 2.xlsx", index=False, float_format="%.1f", sheet_name="Pain data")


##
# ExcelWriter can also be used to append to an existing Excel file:
# with pd.ExcelWriter('output.xlsx',
#                     mode='a') as writer:  
#     df.to_excel(writer, sheet_name='Sheet_name_3')

