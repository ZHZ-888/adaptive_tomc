import pandas as pd

#%%
df_main = pd.read_csv('/home/zzha/PycharmProjects/RampMerging_2026/data/features/rf_at_data_mainlane.csv')
df_ramp = pd.read_csv('/home/zzha/PycharmProjects/RampMerging_2026/data/features/rf_at_data_ramp.csv')
print(df_main.shape, df_ramp.shape)

#%% drop lane with non
df_main_clean = df_main.dropna()
df_ramp_clean = df_ramp.dropna()
print(df_main_clean.shape, df_ramp_clean.shape)

#%% combine main and ramp data
df_mr_clean = pd.concat([df_main_clean, df_ramp_clean], ignore_index=True)
print(df_mr_clean.shape)
print(df_mr_clean.columns)
# ['leader_id', 'record_index', 'prediction_ts', 'platoon_type', 'dis_to_pv',
# 'speed_leader', 'remain_dis_leader', 'm', 'arrival_ts']

#%% modify AHHHH => length
df_mr_clean['platoon_type'] = df_mr_clean['platoon_type'].str.len()

#%% get target (df['arrival_ts'] - df['prediction_ts'])
df_mr_clean['target'] = df_mr_clean['arrival_ts'] - df_mr_clean['prediction_ts']
print(df_mr_clean.shape)
print(df_mr_clean.columns)

#%% save data
df_mr_clean.to_csv('/home/zzha/PycharmProjects/RampMerging_2026/data/features/rf_at_data_rm.csv', index=False)


#%% 260319
df = pd.read_csv('/home/zzha/PycharmProjects/RampMerging_2026/data/features/df_rf_at_260318_raw.csv')
print(df.shape)
df_clean = df.dropna().copy()
df_clean['platoon_type'] = df_clean['platoon_type'].str.len()
df_clean['target'] = df_clean['arrival_ts'] - df_clean['prediction_ts']
print(df_clean)
df_clean.to_csv('/home/zzha/PycharmProjects/RampMerging_2026/data/features/df_rf_at_260318.csv', index=False)