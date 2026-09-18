'''
Based on random_forest_predition_5features2_mr.py
use ndarray to replace dataframe

combine mainline platoon features and ramp platoon features
add a column 'm' => True(1)/False(0)
train RF-based prediction model
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib

#%%
# 1. Data
# Feature data (platoon_type, distance, speed)
# path = '/home/zzha/PycharmProjects/RampMerging4_250208/data/features/df_combined_mr_4f_241128.csv'
# path = '/home/zzha/PycharmProjects/RampMerging_2026/data/features/rf_at_data_rm.csv'
path = '/home/zzha/PycharmProjects/RampMerging_2026/data/features/df_rf_at_260318.csv'
df_ft = pd.read_csv(path)
print(df_ft.columns)
print(len(df_ft))

#%%
# 2. Separate features and target variable
# feature_cols = ['platoon_type', 'dis_to_pv', 'speed_leader', 'remain_dis_leader', 'm']
feature_cols = ['platoon_type', 'leader_to_pv_dis', 'leader_speed', 'leader_left_dis', 'm']

X = df_ft[feature_cols].values
y = df_ft['target'].values

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# 4. Create and train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=800, random_state=36)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

#%%
# 6. Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}") # RMSE^2
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Percentate Error: {mape:.4f}")
print(f"R-squared Score: {r2:.4f}")

#%% 7. Feature importance analysis
importances = model.feature_importances_
feature_names = feature_cols
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

#%% 8. Predict with new data example
ls_features = [7, 221.53, 21.76, 5.73, 0]  # [3, 155.23, 24.74, 53.7524], [7, 127.04, 24.56, 5.98]
new_data = np.array([ls_features])  # Platoon type 1, distance 500 meters, speed 15 m/s
predicted_time = model.predict(new_data)
print(f"\nPredicted Arrival Time for new data: {predicted_time[0]:.2f} seconds")

#%% save the model
# joblib.dump(model, '/home/zzha/PycharmProjects/RampMerging_2026/rf_models/mr_arrival_prediction_model260319_ndarray.pkl')

#%% load model
# loaded_model = joblib.load('/home/zzha/PycharmProjects/RampMerging4_250208/models/mr_arrival_prediction_model241128.pkl')
loaded_model = joblib.load('/home/zzha/PycharmProjects/RampMerging_2026/rf_models/mr_arrival_prediction_model260319_ndarray.pkl')

ls_new_features = [4, 83.05, 24.04, 184.59, 0] # [3, 155.23, 24.74, 53.7524]
ls_new_features =  [17, 96.22, 16.63, 799.32, 1]
new_data = pd.DataFrame([ls_new_features],
                        columns=['platoon_type', 'dis_leader_pv', 'leader_v', 'leader_r_dis', 'm'])
predicted_time = loaded_model.predict(new_data)
print(f"\nPredicted Arrival Time for new data: {predicted_time[0]:.2f} seconds")

#%% avoid build dataframe
ls_new_features = [4, 83.05, 24.04, 184.59, 0]
features = np.array([ls_new_features])
predicted_time = loaded_model.predict(new_data)
print(predicted_time)
print(f"\nPredicted Arrival Time for new data: {predicted_time[0]:.2f} seconds")

#%% save data
data = np.column_stack((y_pred, y_test))
print(data)
np.savetxt('rf_arrival_time.csv',
    data,
    delimiter=",",
    header="y_pred,y_test",
    comments=""
)

#%% Plot
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Create figure and axis
fig, ax = plt.subplots(figsize=(5, 4), dpi=600)

textstr = (
    f"{'RMSE':<4}: 1.693 s\n"
    f"{'MAE':<4}: 0.792 s\n"
    f"{'R²':<4}: 0.973  "
)

ax.text(
    0.95, 0.05,
    textstr,
    transform=ax.transAxes,
    fontsize=10,
    fontfamily='monospace',   # ✅ 核心
    horizontalalignment='right',
    verticalalignment='bottom',
    # bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

# Scatter plot: predicted vs true values
ax.scatter(
    y_pred,
    y_test,
    s=25,
    edgecolor='blue',
    facecolor='none',
    label='Prediction'
)

# Linear regression (first-order polynomial fit)
coefficients = np.polyfit(y_pred, y_test, 1)
fit_line = np.poly1d(coefficients)

# Determine plotting range automatically
min_val = min(y_pred.min(), y_test.min())
max_val = max(y_pred.max(), y_test.max())
upper = np.ceil(max_val / 10) * 10  # Round up to nearest 10 for better visualization

ax.set_xlim(0, 55)
ax.set_ylim(0, 55)

ax.set_xticks([0, 10, 20, 30, 40, 50, 55])
ax.set_yticks([0, 10, 20, 30, 40, 50, 55])

# Generate fitted line
x_fit = np.linspace(min_val, max_val, 600)
y_fit = fit_line(x_fit)

# Plot fitted line
ax.plot(x_fit, y_fit, 'r', label='Linear fit')

# Automatic margins for better visualization
ax.margins(0.05)

# Axis labels
ax.set_xlabel('Predicted value (s)')
ax.set_ylabel('True value (s)')

# Legend
ax.legend()

# Save figure as a PDF (vector format, suitable for publications)
plt.savefig(
    "/home/zzha/PycharmProjects/RampMerging_2026/figures/rf_arrival_times.pdf",
    format="pdf",
    bbox_inches="tight"
)
# plt.savefig("/home/zzha/PycharmProjects/RampMerging_2026/figures/rf_arrival_times.svg",
#             format="svg", bbox_inches="tight")

# Display the figure
plt.show()


#%% Add a zoomed inset
ax_inset = inset_axes(plt.gca(), width="35%", height="35%", loc='lower right', borderpad=2)  # Adjust size and location
ax_inset.scatter(y_pred, y_test, s=30, edgecolor='blue', facecolor='none')  # Replot the scatter points in the inset
ax_inset.plot([10, 14], [10, 14], 'r-', label='Diagonal line')  # Diagonal in zoomed view
ax_inset.plot(x_fit, y_fit, 'r')  # Fit curve in zoomed view

# Set limits for the inset
ax_inset.set_xlim(10, 14)
ax_inset.set_ylim(10, 14)
# ax_inset.grid(True)
# Synchronize ticks with the scripts plot
# Customize ticks for the inset
ax_inset.set_xticks(np.arange(10, 15, 2))  # Custom x-axis ticks (5 to 15 with step 2)
ax_inset.set_yticks(np.arange(10, 15, 2))  # Custom y-axis ticks (5 to 15 with step 2)

# figure
plt.show()

