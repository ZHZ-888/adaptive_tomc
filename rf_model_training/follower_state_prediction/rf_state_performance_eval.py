import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib



#%%
# load data
path = ('/home/zzha/PycharmProjects/RampMerging_2026/data/features/'
        'df_rf_state_mixedHV_260829.csv')
df = pd.read_csv(path)

# data process
# drop None
df_filtered = df.dropna()
# drop dic_leader_to_mcz <= 0
df_filtered2 = df_filtered[df_filtered['dis_leader_to_mcz'] > 0]
# convert state string into '0' or '1'; following_mode = 1, free_mode = 0
df_filtered3 = df_filtered2.copy()
df_filtered3['state'] = df_filtered3['state'].replace({'following_mode': 1, 'free_mode': 0})
print(len(df_filtered3))
print(f'all follower count: {len(df_filtered3)}')


# remove AV
df_hv = df_filtered3.loc[~df_filtered3['follower_id'].str.contains('_av')].copy()
print(f'hv count: {len(df_hv)}')

# split cons, mean, agg
df_cons = df_hv.loc[df_hv['follower_id'].str.contains('_cons')].copy()
print(f'cons count: {len(df_cons)}')
df_mean = df_hv.loc[df_hv['follower_id'].str.contains('_mean')].copy()
print(f'mean count: {len(df_mean)}')
df_agg = df_hv.loc[df_hv['follower_id'].str.contains('_agg')].copy()
print(f'agg count: {len(df_agg)}')

#%% use numpy as input
data_hv = df_hv.to_numpy()
data_cons = df_cons.to_numpy()
data_mean = df_mean.to_numpy()
data_agg = df_agg.to_numpy()

#%% load model
loaded_model = joblib.load('/home/zzha/PycharmProjects/RampMerging_2026/rf_models/'
                           'follower_state_prediction_model_260829_ndarray_final.pkl')

#%% evaluate model
def evaluate_model(data, model, name="Dataset"):
    # split features and target
    X = data[:, 1:5].astype(float)
    y = data[:, 5].astype(int)

    # prediction
    y_pred = model.predict(X)

    # metrics
    print(f"\n{'='*20} {name} {'='*20}")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("Recall:", recall_score(y, y_pred))
    print("F1 Score:", f1_score(y, y_pred))

    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    return y_pred

#%% do
evaluate_model(data_hv, loaded_model, name="HV")
evaluate_model(data_cons, loaded_model, name="cons")
evaluate_model(data_mean, loaded_model, name="mean")
evaluate_model(data_agg, loaded_model, name="agg")
