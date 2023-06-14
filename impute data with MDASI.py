import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# df_init = pd.read_excel("MDASI-v3.xlsx")
# rows_dropped = "none"

# df_init = pd.read_excel("MDASI-v3-removeNoMDASI.xlsx")
# rows_dropped = "blanks"

# df_init = pd.read_excel("df_Fatemeh_merged_ananomyzed_06102023.xlsx", sheet_name="Scores_and_demographics")

col_names = [
"Gender", "HPV/P16 (1= positive, 0=negative, NA=unknown)", "Age", "Smoking", "Alcohol", "Drugs",
# "general_activity_baseline", "general_activity_end_of_xrt", "general_activity_start_of_xrt", "general_activity_week_2", "general_activity_week_3", "general_activity_week_4", "general_activity_week_5", "general_activity_week_6"
# "average_hn_score_baseline", "average_hn_score_end_of_xrt", "average_hn_score_start_of_xrt", "average_hn_score_week_2", "average_hn_score_week_3" "average_hn_score_week_4", "average_hn_score_week_5", "average_hn_score_week_6",
# "mdasi_core_score_baseline", "mdasi_core_score_end_of_xrt", "mdasi_core_score_start_of_xrt", "mdasi_core_score_week_2", "mdasi_core_score_week_3", "mdasi_core_score_week_4", "mdasi_core_score_week_5", "mdasi_core_score_week_6",
# "mdasi_interference_score_baseline", "mdasi_interference_score_end_of_xrt", "mdasi_interference_score_start_of_xrt", "mdasi_interference_score_week_2", "mdasi_interference_score_week_3", "mdasi_interference_score_week_4", "mdasi_interference_score_week_5", "mdasi_interference_score_week_6",
# "mdasi_score_all_baseline", "mdasi_score_all_end_of_xrt", "mdasi_score_all_start_of_xrt", "mdasi_score_all_week_2", "mdasi_score_all_week_3", "mdasi_score_all_week_4", "mdasi_score_all_week_5", "mdasi_score_all_week_6",
"mdasi_appetite_baseline", "mdasi_appetite_end_of_xrt", "mdasi_appetite_start_of_xrt", "mdasi_appetite_week_2", "mdasi_appetite_week_3", "mdasi_appetite_week_4", "mdasi_appetite_week_5", "mdasi_appetite_week_6",
"mdasi_choke_baseline", "mdasi_choke_end_of_xrt", "mdasi_choke_start_of_xrt", "mdasi_choke_week_2", "mdasi_choke_week_3", "mdasi_choke_week_4", "mdasi_choke_week_5", "mdasi_choke_week_6",
"mdasi_constipation_baseline", "mdasi_constipation_end_of_xrt", "mdasi_constipation_start_of_xrt", "mdasi_constipation_week_2", "mdasi_constipation_week_3", "mdasi_constipation_week_4", "mdasi_constipation_week_5", "mdasi_constipation_week_6",
"mdasi_distress_baseline", "mdasi_distress_end_of_xrt", "mdasi_distress_start_of_xrt", "mdasi_distress_week_2", "mdasi_distress_week_3", "mdasi_distress_week_4", "mdasi_distress_week_5", "mdasi_distress_week_6",
"mdasi_drowsy_baseline", "mdasi_drowsy_end_of_xrt", "mdasi_drowsy_start_of_xrt", "mdasi_drowsy_week_2", "mdasi_drowsy_week_3", "mdasi_drowsy_week_4", "mdasi_drowsy_week_5", "mdasi_drowsy_week_6",
"mdasi_drymouth_baseline", "mdasi_drymouth_end_of_xrt", "mdasi_drymouth_start_of_xrt", "mdasi_drymouth_week_2", "mdasi_drymouth_week_3", "mdasi_drymouth_week_4", "mdasi_drymouth_week_5", "mdasi_drymouth_week_6",
"mdasi_fatigue_baseline", "mdasi_fatigue_end_of_xrt", "mdasi_fatigue_start_of_xrt", "mdasi_fatigue_week_2", "mdasi_fatigue_week_3", "mdasi_fatigue_week_4", "mdasi_fatigue_week_5", "mdasi_fatigue_week_6",
"mdasi_memory_baseline", "mdasi_memory_end_of_xrt", "mdasi_memory_start_of_xrt", "mdasi_memory_week_2", "mdasi_memory_week_3", "mdasi_memory_week_4", "mdasi_memory_week_5", "mdasi_memory_week_6",
"mdasi_mucositis_baseline", "mdasi_mucositis_end_of_xrt", "mdasi_mucositis_start_of_xrt", "mdasi_mucositis_week_2", "mdasi_mucositis_week_3", "mdasi_mucositis_week_4", "mdasi_mucositis_week_5", "mdasi_mucositis_week_6",
"mdasi_mucus_baseline", "mdasi_mucus_end_of_xrt", "mdasi_mucus_start_of_xrt", "mdasi_mucus_week_2", "mdasi_mucus_week_3", "mdasi_mucus_week_4", "mdasi_mucus_week_5", "mdasi_mucus_week_6",
"mdasi_nausea_baseline", "mdasi_nausea_end_of_xrt", "mdasi_nausea_start_of_xrt", "mdasi_nausea_week_2", "mdasi_nausea_week_3", "mdasi_nausea_week_4", "mdasi_nausea_week_5", "mdasi_nausea_week_6",
"mdasi_numb_baseline", "mdasi_numb_end_of_xrt", "mdasi_numb_start_of_xrt", "mdasi_numb_week_2", "mdasi_numb_week_3", "mdasi_numb_week_4", "mdasi_numb_week_5", "mdasi_numb_week_6",
"mdasi_pain_baseline", "mdasi_pain_end_of_xrt", "mdasi_pain_start_of_xrt", "mdasi_pain_week_2", "mdasi_pain_week_3", "mdasi_pain_week_4", "mdasi_pain_week_5", "mdasi_pain_week_6",
"mdasi_sad_baseline", "mdasi_sad_end_of_xrt", "mdasi_sad_start_of_xrt", "mdasi_sad_week_2", "mdasi_sad_week_3", "mdasi_sad_week_4", "mdasi_sad_week_5", "mdasi_sad_week_6",
"mdasi_skin_baseline", "mdasi_skin_end_of_xrt", "mdasi_skin_start_of_xrt", "mdasi_skin_week_2", "mdasi_skin_week_3", "mdasi_skin_week_4", "mdasi_skin_week_5", "mdasi_skin_week_6",
"mdasi_sleep_baseline", "mdasi_sleep_end_of_xrt", "mdasi_sleep_start_of_xrt", "mdasi_sleep_week_2", "mdasi_sleep_week_3", "mdasi_sleep_week_4", "mdasi_sleep_week_5", "mdasi_sleep_week_6",
"mdasi_sob_baseline", "mdasi_sob_end_of_xrt", "mdasi_sob_start_of_xrt", "mdasi_sob_week_2", "mdasi_sob_week_3", "mdasi_sob_week_4", "mdasi_sob_week_5", "mdasi_sob_week_6",
"mdasi_swallow_baseline", "mdasi_swallow_end_of_xrt", "mdasi_swallow_start_of_xrt", "mdasi_swallow_week_2", "mdasi_swallow_week_3", "mdasi_swallow_week_4", "mdasi_swallow_week_5", "mdasi_swallow_week_6",
"mdasi_taste_baseline", "mdasi_taste_end_of_xrt", "mdasi_taste_start_of_xrt", "mdasi_taste_week_2", "mdasi_taste_week_3", "mdasi_taste_week_4", "mdasi_taste_week_5", "mdasi_taste_week_6",
"mdasi_teeth_baseline", "mdasi_teeth_end_of_xrt", "mdasi_teeth_start_of_xrt", "mdasi_teeth_week_2", "mdasi_teeth_week_3", "mdasi_teeth_week_4", "mdasi_teeth_week_5", "mdasi_teeth_week_6",
"mdasi_voice_baseline", "mdasi_voice_end_of_xrt", "mdasi_voice_start_of_xrt", "mdasi_voice_week_2", "mdasi_voice_week_3", "mdasi_voice_week_4", "mdasi_voice_week_5", "mdasi_voice_week_6",
"mdasi_vomit_baseline", "mdasi_vomit_end_of_xrt", "mdasi_vomit_start_of_xrt", "mdasi_vomit_week_2", "mdasi_vomit_week_3", "mdasi_vomit_week_4", "mdasi_vomit_week_5", "mdasi_vomit_week_6"
]
# df = df_init[col_names]
# df = df.iloc[:,0:38]
# print(list(df.columns))

# df = df.replace("NA", np.nan)
# print(df)

# ### Initiate imputer with estimator
# # lr = LinearRegression()
# # imp = IterativeImputer(missing_values=np.nan, max_iter=100, verbose=2, imputation_order='ascending',
# #                        random_state=0, estimator=lr, min_value=0, max_value=10)  # default estimator is BayesianRidge()
# # method = "LR"

# br = BayesianRidge()
# imp = IterativeImputer(missing_values=np.nan, max_iter=100, verbose=2, imputation_order='ascending',
#                        random_state=0, estimator=br, min_value=0, max_value=10)  # default estimator is BayesianRidge()
# # method = "BR"

# df_array = imp.fit_transform(df)
# # print(df_array.size)
# # print(df_array)

# ### Convert to a DataFrame
# df_final = pd.DataFrame(data = df_array, index = range(df_array.shape[0]), columns=df.columns)
# df_final = df_final.round(decimals=0)
# df_final = df_final.astype(int)
# # Fix data
# df_final["Gender"] = df_final["Gender"].replace([2,3,4,5,6,7,8,9,10], 1)
# df_final["HPV/P16 (1= positive, 0=negative, NA=unknown)"] = df_final["HPV/P16 (1= positive, 0=negative, NA=unknown)"].replace([2,3,4,5,6,7,8,9,10], 1)
# df_final["Age"] = df_final["Age"].replace([2,3,4,5,6,7,8,9,10], 1)
# df_final["Smoking"] = df_final["Smoking"].replace([3,4,5,6,7,8,9,10], 2)
# df_final["Drugs"] = df_final["Drugs"].replace([2,3,4,5,6,7,8,9,10], 1)
# # convert_dict = {"Column name": int, "Column name": int,"Pain_score_W2": int}
# # df_final = df_final.astype(convert_dict)
# print(df_final)

# ### Save to excel files (using 100 iterations and 'ascending' imputation order)
# df_final.to_excel(f"MDASI imputed {method} {rows_dropped} dropped.xlsx", index=False, float_format="%.1f", sheet_name="MDASI scores")
# # df_final.to_excel("MDASI imputed LR none dropped.xlsx", index=False, float_format="%.1f", sheet_name="MDASI scores")
# # df_final.to_excel("MDASI imputed LR blanks dropped.xlsx", index=False, float_format="%.1f", sheet_name="MDASI scores")
# df_final.to_excel("MDASI imputed BR none dropped.xlsx", index=False, float_format="%.1f", sheet_name="MDASI scores")
# # df_final.to_excel("MDASI imputed BR blanks dropped.xlsx", index=False, float_format="%.1f", sheet_name="MDASI scores")


##########################################################################
##########################################################################
filenames = ["MDASI-v3.xlsx", "MDASI-v3-removeNoMDASI.xlsx"]
rows_dropped = ["none", "blanks"]

# df_init = pd.read_excel("MDASI-v3-removeNoMDASI.xlsx")
# dropped = "blanks"

# Execute in for loop
estimators = [
    # BayesianRidge(),
    # LinearRegression(),
    RandomForestRegressor(
        n_estimators=50,  # default 100
        # max_depth=10,
        # bootstrap=True,
        # max_samples=0.5,
        # n_jobs=2,
        random_state=0,
    ),
    # KNeighborsRegressor(n_neighbors=15),
]

methods = [
    # "LR",
    # "BR",
    "RF",
    # "KNN"
    ]

num_iterations = [
    # 100,
    # 100,
    25,
    # 100
    ]

for file, rd in zip(filenames, rows_dropped):
    df_init = pd.read_excel(file)
    df = df_init[col_names]
    df = df.replace("NA", np.nan)

    for impute_estimator, method, num_iter in zip(estimators, methods, num_iterations):
        imp = IterativeImputer(missing_values=np.nan, max_iter=num_iter, verbose=2, imputation_order='ascending',
                        random_state=0, estimator=impute_estimator, min_value=0, max_value=10)
        df_array = imp.fit_transform(df)

        df_final = pd.DataFrame(data = df_array, index = range(df_array.shape[0]), columns=df.columns)
        df_final = df_final.round(decimals=0)
        df_final = df_final.astype(int)
        
        df_final["Gender"] = df_final["Gender"].replace([2,3,4,5,6,7,8,9,10], 1)
        df_final["HPV/P16 (1= positive, 0=negative, NA=unknown)"] = df_final["HPV/P16 (1= positive, 0=negative, NA=unknown)"].replace([2,3,4,5,6,7,8,9,10], 1)
        df_final["Age"] = df_final["Age"].replace([2,3,4,5,6,7,8,9,10], 1)
        df_final["Smoking"] = df_final["Smoking"].replace([3,4,5,6,7,8,9,10], 2)
        df_final["Drugs"] = df_final["Drugs"].replace([2,3,4,5,6,7,8,9,10], 1)

        df_final.to_excel(f"MDASI imputed {method} {rd} dropped.xlsx", index=False, float_format="%.1f", sheet_name="MDASI scores")
