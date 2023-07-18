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
folder = "Policy bmi"
filename = "rewards_all_weeks_BMI.xlsx"
fullpath_read = os.path.join(work_dir, folder, filename)

# col_ranges = ["B2:D21", "G2:I21", "L2:N21", "B26:D45", "G26:I45", "L26:N45"]
col_ranges = ["B:D", "G:I", "L:N","B:D", "G:I", "L:N"]
week = 0
for col_range in col_ranges:
    # df_init = pd.read_excel(fullpath_read, skiprows=1, usecols=col_range)
    week += 1
    if week < 4:
        df_init = pd.read_excel(fullpath_read, header=None, names = [1,2,3], skiprows=1, nrows=20, usecols=col_range)
    else:
        df_init = pd.read_excel(fullpath_read, header=None, names = [1,2,3], skiprows=25, nrows = 20, usecols=col_range)

    df = df_init.drop(labels=[6,13], axis=0)
    # print(df)

    ### Interpolate
    estimator = LinearRegression()
    method = "LR" # "BR"

    num_iter = 100
    imp = IterativeImputer(missing_values=np.nan, max_iter=num_iter, verbose=2, imputation_order='ascending',
                        random_state=0, estimator=estimator)

    df_array = imp.fit_transform(df)

    df_final = pd.DataFrame(data = df_array, index = range(df_array.shape[0]), columns=df.columns)
    df_final.index = ["[0,MM,G]", "[0,MM,A]", "[0,MM,P]", "[0,MS,G]", "[0,MS,A]", "[0,MS,P]", \
        "[1,MM,G]", "[1,MM,A]", "[1,MM,P]", "[1,MS,G]", "[1,MS,A]", "[1,MS,P]", \
        "[2,MM,G]", "[2,MM,A]", "[2,MM,P]", "[2,MS,G]", "[2,MS,A]", "[2,MS,P]"]
    # print(df_final)

    ### Save to excel files (using 100 iterations and 'ascending' imputation order)
    fullpath_write = os.path.join(work_dir, folder, f"Rewards BMI imputed method {method}.xlsx")
    if week == 1:
        df_final.to_excel(fullpath_write, index=True, float_format="%.7f", sheet_name=f"Week {week}")
    else:
        with pd.ExcelWriter(fullpath_write, mode='a') as writer:  
            df_final.to_excel(writer, index=True, float_format="%.7f", sheet_name=f"Week {week}")
