'''
update, combine ramp and scripts features, add a new column 'm' => 1(True), 0(False)
241122, new platoon type, 1 lead 5 and 1 lead 6,
new av_p, 0.1, 0.15, 0.2, 0.3
But, only ramp or only scripts features
'''
# organising data

import pandas as pd

# new mainline data
# /home/zzha/PycharmProjects/RampMerging3/data/df_mft_{av_p}_1122.csv
# df_ft_m1 = pd.read_csv('/home/zzha/PycharmProjects/RampMerging3/data/df_mft_0.1_1122.csv')
# df_ft_m2 = pd.read_csv('/home/zzha/PycharmProjects/RampMerging3/data/df_mft_0.15_1122.csv')
# df_ft_m3 = pd.read_csv('/home/zzha/PycharmProjects/RampMerging3/data/df_mft_0.2_1122.csv')
# df_ft_m4 = pd.read_csv('/home/zzha/PycharmProjects/RampMerging3/data/df_mft_0.3_1122.csv')

df_ft_m1 = pd.read_csv('/home/zzha/PycharmProjects/RampMerging3/data/df_mft_0.1_1128_lp.csv')
df_ft_m2 = pd.read_csv('/home/zzha/PycharmProjects/RampMerging3/data/df_mft_0.15_1128_lp.csv')
df_ft_m3 = pd.read_csv('/home/zzha/PycharmProjects/RampMerging3/data/df_mft_0.2_1128_lp.csv')
df_ft_m4 = pd.read_csv('/home/zzha/PycharmProjects/RampMerging3/data/df_mft_0.3_1128_lp.csv')

# ramp data
df_ft_r1 = pd.read_csv('/home/zzha/PycharmProjects/RampMerging3/data/df_rft_0.1_1128.csv')
df_ft_r2 = pd.read_csv('/home/zzha/PycharmProjects/RampMerging3/data/df_rft_0.15_1128.csv')
df_ft_r3 = pd.read_csv('/home/zzha/PycharmProjects/RampMerging3/data/df_rft_0.2_1128.csv')
df_ft_r4 = pd.read_csv('/home/zzha/PycharmProjects/RampMerging3/data/df_rft_0.3_1128.csv')

#%%
# clean data
# 4 features (leader_pv_id, leader_v, leader_r_dis, platoon_type), 1 target (tail_r_t)
# 80% training data, 20% test data
# combination 1, 4 features
def filter_data(df):
    df_filtered = df.loc[:, ['platoon_type', 'dis_leader_pv', 'leader_v', 'leader_r_dis', 'tail_arr_duration']]
    return df_filtered

# combination 2, 6 features
def filter_data2(df):
    df_filtered = df.loc[:, ['platoon_type', 'dis_leader_pv', 'leader_v', 'leader_r_dis',
                             'tail_arr_duration', 'tail_v', 'tail_r_dis']]
    return df_filtered

#%%
# clean, combine and shuffle
df_ft_m1f = filter_data2(df_ft_m1)
df_ft_m2f = filter_data2(df_ft_m2)
df_ft_m3f = filter_data2(df_ft_m3)
df_ft_m4f = filter_data2(df_ft_m4)

df_ft_r1f = filter_data2(df_ft_r1)
df_ft_r2f = filter_data2(df_ft_r2)
df_ft_r3f = filter_data2(df_ft_r3)
df_ft_r4f = filter_data2(df_ft_r4)

df_combined_m = pd.concat([df_ft_m1f, df_ft_m2f, df_ft_m3f, df_ft_m4f], ignore_index=True)
df_combined_r = pd.concat([df_ft_r1f, df_ft_r2f, df_ft_r3f, df_ft_r4f], ignore_index=True)

#%%
# add new columns (m is true/1 or false/0) for features
df_combined_m['m'] = 1
df_combined_r['m'] = 0
# and combine m and r
df_combined = pd.concat([df_combined_m, df_combined_r], ignore_index=True)

#%%
# shift the format of platoon_type: from 'AHHH' to its length
df_combined_c = df_combined.copy()
df_combined_c['platoon_type'] = df_combined_c['platoon_type'].str.len()
df_combined_c
df_combined_c.to_csv('data/df_combined_mr_4f_241128.csv', index=False)

#%% plot 2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Extracting columns for plotting
x = df_combined_c['dis_leader_pv']
y = df_combined_c['leader_v']
z = df_combined_c['leader_r_dis']
c = df_combined_c['tail_arr_duration']  # Color intensity
s = df_combined_c['platoon_type'] # Color intensity

# Create 3D scatter plot
fig = plt.figure(dpi=600)
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color based on tail_arr_duration
sc = ax.scatter(x, y, s, s=0.2, c=c, cmap='coolwarm')

# Adding color bar
color_bar = plt.colorbar(sc, ax=ax, shrink=0.7, aspect=15, location='left')
color_bar.set_label('Tail Arrival Duration (s)')

# Labeling axes
ax.set_xlabel('dis_leader_pv (m)')
ax.set_ylabel('leader_v (m/s)')
ax.set_zlabel('leader_r_dis (m)')

# Show plot
plt.show()

#%% ploty
import plotly.express as px
# Create 3D scatter plot with Plotly
import plotly.io as pio
pio.renderers.default = 'browser'
fig = px.scatter_3d(df_combined_c,
                    x='dis_leader_pv',
                    y='leader_v',
                    z='leader_r_dis',
                    color='tail_arr_duration',  # Color based on duration
                    # size='platoon_type',        # Point size based on platoon_type
                    # color_continuous_scale='coolwarm',
                    labels={
                        'dis_leader_pv': 'dis_leader_pv (m)',
                        'leader_v': 'leader_v (m/s)',
                        'leader_r_dis': 'leader_r_dis (m)',
                        'tail_arr_duration': 'Tail Arrival Duration (s)'
                    })

# Show plot
fig.show()