'''
FS_random_forest_251216.py
avoid dataframe
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # FOR CLASSIFICATION
# FOR CLASSIFICATION
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib


#%% data preprocessing
def prepare_xy(df):
    # drop None
    df_filtered = df.dropna()

    # drop line if dis_leader_to_mcz < 0
    df_filtered = df_filtered.loc[
        df_filtered['dis_leader_to_mcz'] >= 0
    ].copy()

    # convert state string into 0/1
    df_filtered['state'] = df_filtered['state'].replace({
        'following_mode': 1,
        'free_mode': 0
    })

    # dataframe -> numpy
    data = df_filtered.to_numpy()

    # features and target
    X = data[:, 1:5].astype(float)
    y = data[:, 5].astype(int)

    return X, y

#%% input data
# path = '/home/zzha/PycharmProjects/RampMerging_2026/data/features/df_rf_state_mixedHV_260829_homo_Kruass_seed21_50.csv'
path = '/home/zzha/PycharmProjects/RampMerging_2026/data/features/df_rf_state_homoHV_seed21-60_avp102030_260905.csv'
df_train = pd.read_csv(path)
print(df_train.columns)
print(len(df_train))

path_test = '/home/zzha/PycharmProjects/RampMerging_2026/data/features/df_rf_state_mixedHV_260829_homo_Kruass_seed51_60.csv'
df_test = pd.read_csv(path_test)
print(df_test.columns)
print(len(df_test))

#%%
X_train, y_train = prepare_xy(df_train)
X_test, y_test = prepare_xy(df_test)

#%% Create and train the Random Forest Regressor model
model = RandomForestClassifier(n_estimators=500, random_state=36)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)
#%%
# 6. Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Feature importance analysis
importances = model.feature_importances_
feature_names = ["v_leader", "dis_leader_to_mcz", "n_veh_between", "time_headway_to_leader"]
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

#%% save the model
# joblib.dump(model, '/home/zzha/PycharmProjects/RampMerging_2026/rf_models/'
#                    'follower_state_prediction_model_260829_ndarray_final.pkl')

joblib.dump(model, '/home/zzha/PycharmProjects/RampMerging_2026/rf_models/'
                   'follower_state_prediction_model_260905_ndarray_updateAVtau.pkl')

#%% load model
loaded_model = joblib.load('/home/zzha/PycharmProjects/RampMerging_2026/rf_models/'
                           'follower_state_prediction_model_260715_ndarray.pkl')
ls_new_features = [25, 778, 3, 12.7]

X_new = np.array(ls_new_features, dtype=float).reshape(1, -1)
new_state = loaded_model.predict(X_new)
print(new_state)

#%% plot
# Import necessary libraries
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

main_blue = "#0000FF"
custom_blues = LinearSegmentedColormap.from_list(
    'custom_blues',
    ['white', main_blue]
)

# Convert predicted probabilities to binary class (0 or 1)
y_pred_class = (y_pred >= 0.5).astype(int)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred_class)

# Plot the confusion matrix with a color map
fig, ax = plt.subplots(figsize=(5, 4), dpi=600)

textstr = (
    f"{'Precision':<9}: 0.962\n"
    f"{'Recall':<9}: 0.981\n"
    f"{'F1-score':<9}: 0.979"
)

ax.text(
    0.97, 0.03,
    textstr,
    transform=ax.transAxes,
    fontsize=10,
    # color='white',
    fontfamily='monospace',
    horizontalalignment='right',
    verticalalignment='bottom',
    bbox=dict(boxstyle='round', facecolor='white', alpha=1)
)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=custom_blues, ax=ax)  # You can also try 'Greens', 'Purples', 'Oranges', etc.

# save as pdf
# plt.savefig("/home/zzha/PycharmProjects/RampMerging_2026/figures/rf_following_states1.pdf",
#             format='pdf', bbox_inches='tight')
# plt.savefig("/home/zzha/PycharmProjects/RampMerging_2026/figures/rf_following_states.svg", format="svg", bbox_inches="tight")
plt.show()

#%%
# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LinearSegmentedColormap

# Custom colormap
main_blue = "#0000FF"
custom_blues = LinearSegmentedColormap.from_list(
    "custom_blues",
    ["white", main_blue]
)

# Convert predicted probabilities to binary class
# If y_pred is already class labels, replace this line with: y_pred_class = y_pred
y_pred_class = (y_pred >= 0.5).astype(int)

# Raw confusion matrix
cm = confusion_matrix(y_test, y_pred_class, labels=[0, 1])

# Row-normalized confusion matrix
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# Class labels
class_names = ["Free", "Coupled-following"]

# Plot
fig, ax = plt.subplots(figsize=(5, 4), dpi=600)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_norm,
    display_labels=class_names
)

disp.plot(
    cmap=custom_blues,
    ax=ax,
    values_format=".1%",
    colorbar=True
)

# Axis labels
ax.set_xlabel("Predicted label", fontsize=11)
ax.set_ylabel("True label", fontsize=11)
ax.set_title("Normalized confusion matrix", fontsize=12)

# Improve tick label style
ax.tick_params(axis="both", labelsize=10)

# Make text color readable
for text in disp.text_.ravel():
    value = float(text.get_text().replace("%", ""))
    text.set_fontsize(11)
    text.set_color("white" if value > 50 else "black")

plt.tight_layout()

# Save as pdf/svg if needed
# plt.savefig("/home/zzha/PycharmProjects/RampMerging_2026/figures/rf_following_states_norm.pdf",
#             format="pdf", bbox_inches="tight")
# plt.savefig("/home/zzha/PycharmProjects/RampMerging_2026/figures/rf_following_states_norm.svg",
#             format="svg", bbox_inches="tight")

plt.show()
