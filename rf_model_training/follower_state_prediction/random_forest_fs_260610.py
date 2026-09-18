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

#%% input data
# path = '/home/zzha/PycharmProjects/RampMerging_2026/data/features/df_rf_state_feature_target_260610_2.csv'
path = '/home/zzha/PycharmProjects/RampMerging_2026/data/features/df_rf_state_feature_target_260715.csv'
df = pd.read_csv(path)
print(df.columns)
print(df.head())
print(len(df))

#%% data preprocessing
# drop None
df_filtered = df.dropna()
print(len(df_filtered))

# drop line if dis_leader_to_mcz <0
df_filtered2 = df_filtered[df_filtered['dis_leader_to_mcz'] >= 0]
print(len(df_filtered2))

# convert state string into '0' or '1'; following_mode = 1, free_mode = 0
df_filtered3 = df_filtered2.copy()
df_filtered3['state'] = df_filtered3['state'].replace({'following_mode': 1, 'free_mode': 0})
# columns: ['follower_id', 'v_leader', 'dis_leader_to_mcz', 'n_veh_between', 'time_headway_to_leader', 'state', 'leader_id']
print(len(df_filtered3))

# use numpy as input
data = df_filtered3.to_numpy()

#%% model training
# 2. Separate features and target variable
X = data[:, 1:5].astype(float)
y = data[:, 5].astype(int) # should be int instead of object

#%%
# 3. Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# undersampling function
def undersample_training_data(X_train, y_train, ratio=2, random_state=42):
    """
    Undersample the majority class in the training set only.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels. 0 = free, 1 = coupled-following.
    ratio : int
        Target ratio of class 1 to class 0.
        ratio=2 means class 0 : class 1 = 1 : 2.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train_balanced : np.ndarray
        Undersampled training features.
    y_train_balanced : np.ndarray
        Undersampled training labels.
    """

    train_df = pd.DataFrame(X_train, columns=[
        'v_leader',
        'dis_leader_to_mcz',
        'n_veh_between',
        'time_headway_to_leader'
    ])
    train_df['state'] = y_train

    train_0 = train_df[train_df['state'] == 0]
    train_1 = train_df[train_df['state'] == 1]

    print("Before undersampling:")
    print(train_df['state'].value_counts())

    n_class1 = min(len(train_1), len(train_0) * ratio)

    train_1_sampled = train_1.sample(
        n=n_class1,
        random_state=random_state
    )

    train_balanced = pd.concat(
        [train_0, train_1_sampled],
        ignore_index=True
    )

    train_balanced = train_balanced.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    print("After undersampling:")
    print(train_balanced['state'].value_counts())

    X_train_balanced = train_balanced[[
        'v_leader',
        'dis_leader_to_mcz',
        'n_veh_between',
        'time_headway_to_leader'
    ]].to_numpy()

    y_train_balanced = train_balanced['state'].to_numpy()

    return X_train_balanced, y_train_balanced

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#%%
# 4. Create and train the Random Forest Regressor model
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
# joblib.dump(model, 'Models/follower_state_prediction_model_250415.pkl')
# joblib.dump(model, '/home/zzha/PycharmProjects/RampMerging_2026/rf_models/follower_state_prediction_model_260610_ndarray.pkl')
joblib.dump(model, '/home/zzha/PycharmProjects/RampMerging_2026/rf_models/'
                   'follower_state_prediction_model_260715_ndarray.pkl')

#%% load model
# loaded_model = joblib.load('Models/follower_state_prediction_model_250415.pkl')
# loaded_model = joblib.load('Models/follower_state_prediction_model_250501.pkl')
# loaded_model = joblib.load('/home/zzha/PycharmProjects/RampMerging_2026/rf_models/follower_state_prediction_model_260610_ndarray.pkl') # employed
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
import numpy as np
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
