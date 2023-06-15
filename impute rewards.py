import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

week = 6
df_init = pd.read_excel(f"Reward_SAT_W{week}.xlsx")
df = df_init.iloc[:-2]
df.drop(columns=df_init.columns[0], axis=1, inplace=True)
# print(df_init)
# print(df_init.columns)

# col_names = []
# df = df_init[col_names]

### Initiate imputer with estimator
estimator = LinearRegression()
method = "LR"

# estimator = BayesianRidge()  # default
# method = "BR"

# kernel = "rbf"  # stopping criteria reached in 14 iter
# kernel = "linear"
# estimator = SVR(kernel=kernel)
# method = f"SVR kernel {kernel}"

num_iter = 100
order = "ascending"
imp = IterativeImputer(missing_values=np.nan, max_iter=num_iter, verbose=2, imputation_order=order,
                       random_state=0, estimator=estimator, min_value=0, max_value=10)

df_array = imp.fit_transform(df)
# print(df_array.size)
# print(df_array)

### Convert to a DataFrame
df_final = pd.DataFrame(data = df_array, index = range(df_array.shape[0]), columns=df.columns)

# ### Save to excel files (using 100 iterations and 'ascending' imputation order)
df_final.to_excel(f"Rewards imputed {method} week {week} order {order}.xlsx", index=False, sheet_name="Rewards") # float_format="%.4f"
