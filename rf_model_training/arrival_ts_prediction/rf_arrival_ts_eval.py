import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib


#%% function: data preprocess
def process_data(df):
    df = df.copy()
    # 1. drop nan
    df = df.dropna()
    # 2. modify 'AHH' to '3'
    df['platoon_type'] = df['platoon_type'].str.len()
    # 3. add target
    df['target'] = df['arrival_ts'] - df['prediction_ts']
    return df

#%% Feature data (platoon_type, distance, speed)
path = '/home/zzha/PycharmProjects/RampMerging_2026/data/features/df_rf_arrival_time_mixedHV_260830.csv'
df_ft = pd.read_csv(path)
print(df_ft.columns)
print(len(df_ft))

#%% Data preprocess
df_ft2 = process_data(df_ft)
print(df_ft2.columns)
print(len(df_ft2))

#%% Separate features and target variable
feature_cols = ['platoon_type', 'leader_to_pv_dis', 'leader_speed', 'leader_left_dis', 'm']

X = df_ft2[feature_cols].values
y = df_ft2['target'].values

#%% Load model
loaded_model = joblib.load('/home/zzha/PycharmProjects/RampMerging_2026/'
                           'rf_models/mr_arrival_prediction_model260319_ndarray.pkl')

#%% Make prediction
y_pred = loaded_model.predict(X)

#%% Evaluate model performance
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
mape = mean_absolute_percentage_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error: {mse:.4f}") # RMSE^2
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Percentate Error: {mape:.4f}")
print(f"R-squared Score: {r2:.4f}")


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

