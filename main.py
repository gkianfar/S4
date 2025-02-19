import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
import matplotlib.patches as mpatches
from collections.abc import Iterable
import os
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import copy
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
import shap
from utils import *
# Argparse
import argparse


# Run python main --if_global --dataset_path "/content/drive/MyDrive/datasets/dataset_clean0_tm0_rec0_th0.03_occ0_seed2025_neg1.csv" --seed 2025 --model_name "SVM" --retrain
# Initialize the argument parser
parser = argparse.ArgumentParser(description="Process some integers.")

# Add the 'if_global' argument as a boolean flag
parser.add_argument(
    '--if_global',
    action='store_true',
    help='Set if_global to True',
)

parser.add_argument(
    '--retrain',
    action='store_true',
    help='Set if you want to remove features and retrain',
)

# Add the 'seed' argument with a default value
parser.add_argument(
    '--seed',
    type=int,
    default=2025,
    help='Set the seed for repeatable train-test-validation split (default: 2025)',
)

parser.add_argument(
    '--dataset_path',
    type=str,
    help='The path of the target dataset'
)

parser.add_argument(
    '--model_name',
    choices=['DecisionTree', 'SVM', 'MLP'],
    help='Specify the model to use. Options: DecisionTree, SVM, MLP.'
)


# Parse the command-line arguments
args = parser.parse_args()

print(args)

# Assign the parsed arguments to variables
if_global = args.if_global
seed = args.seed
model_name = args.model_name
retrain = args.retrain
dataset_path = args.dataset_path
#dataset_path = f'/content/drive/MyDrive/datasets/dataset_{filename}.csv'
ratios = [0.9,0.1] # [train, test]

# Check if the provided dataset_path exists
if not os.path.exists(args.dataset_path):
    raise FileNotFoundError(f"The specified dataset path does not exist: {args.dataset_path}")

# Set directory
#%cd /content/TIHM-Dataset-Visualization
DPATH = './Data'
SAVE_PATH = './Figs/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

dataset = pd.read_csv(dataset_path)

# Extract the base filename without the directory path
f = dataset_path.split('/')[-1]  # 'dataset_example.csv'

# Remove the prefix 'dataset_' and the suffix '.csv'
filename = f[len('dataset_'):-len('.csv')]

print(f'filename: {filename}')

# Encode nonnumeric features and correcting data types

# Modify dtype
dataset['normal_samples'] = dataset['normal_samples'].astype(float)

#Weekdays
ohe_col = 'week_day'
insert_position = dataset.columns.get_loc(ohe_col)
ohe = OneHotEncoder(sparse_output=False)
encoded = ohe.fit_transform(dataset[[ohe_col]])
df_encoded = pd.DataFrame(encoded, columns=ohe.get_feature_names_out([ohe_col]))
dataset = dataset.drop(columns=ohe_col)

dataset = pd.concat([dataset.iloc[:, :insert_position], df_encoded, dataset.iloc[:, insert_position:]], axis=1)
#sorted_unique_week_day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday', 'Saturday', 'Sunday']
#map_week_day = dict(zip(sorted_unique_week_day, np.arange(len(sorted_unique_week_day))/(len(sorted_unique_week_day)-1)))
#dataset['week_day'] = dataset['week_day'].map(map_week_day)

# Gender
map_gender = {'Male':0, 'Female':1}
dataset['gender'] = dataset['gender'].map(map_gender)

#Age group
sorted_unique_age = sorted(dataset.age.unique(), key=lambda x: int(x.strip('()[]').split(', ')[0]))
map_age = dict(zip(sorted_unique_age, np.arange(len(sorted_unique_age))/(len(sorted_unique_age)-1)))
dataset['age'] = dataset['age'].map(map_age)

# Drop tha mean and variance columns
dataset = dataset[dataset.columns[~dataset.columns.str.endswith(('_std', '_mean'))]].copy()


# Dataset partitioning
if if_global:
  dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
  total_samples = dataset.shape[0]
  dataset_train = dataset.iloc[:int(total_samples*ratios[0])]
  dataset_test = dataset.iloc[int(total_samples*(ratios[0])):]
else:
  stat= dataset.groupby('patient_id').agg(
      zeros = ('label', count_zeros),
      ones = ('label', count_ones)
  ).reset_index()
  one_portion = 1/2
  total_samples = dataset.shape[0]
  train_size = int(total_samples*ratios[0])
  test_size = total_samples-train_size
  dataset_train = []
  dataset_test = []
  filled_train = 0
  filles_test = 0

  patient_with_single_one = stat['patient_id'][stat['ones']==1].tolist()
  filled_train += dataset[dataset['patient_id'].isin(patient_with_single_one)].shape[0]
  dataset_train.append(dataset[dataset['patient_id'].isin(patient_with_single_one)].copy().reset_index(drop=True))
  dataset = dataset.drop(dataset[dataset['patient_id'].isin(patient_with_single_one)].index).reset_index(drop=True)

  num_test_patients = int(test_size*one_portion)
  selected_patients = np.random.choice(dataset['patient_id'].unique(), size=num_test_patients, replace=False)
  dataset_train =[]
  dataset_test = []

  for p in selected_patients:

    for l in [0,1]:
      # Get the subset
      subset = dataset[(dataset['label'] == l) & (dataset['patient_id'] == p)]

      if not subset.empty:
          # Randomly select one row
          selected_index = np.random.choice(subset.index)

          # Drop the selected row
          dataset_test.append(dataset.iloc[selected_index])
          dataset = dataset.drop(index=selected_index).reset_index(drop=True)
          filles_test +=1
  dataset_train.append(dataset)
  dataset_train = pd.concat(dataset_train)
  dataset_test = pd.concat(dataset_test, axis=1).T.reset_index(drop=True)


# Normalization
if if_global:
    # Global scaling: using the entire dataset statistics
    scaler = StandardScaler()
    min_max_scaler = MinMaxScaler()

    # Exclude 'normal_samples' column (assuming it's in the columns with index 12 onwards)
    columns_to_normalize = dataset_train.iloc[:, 12:].columns.difference(['normal_samples','label'])

    # Apply normalization only to the selected columns
    dataset_train[columns_to_normalize] = scaler.fit_transform(dataset_train[columns_to_normalize])
    dataset_test[columns_to_normalize] = scaler.transform(dataset_test[columns_to_normalize])

else:
    min_max_scaler = MinMaxScaler()
    # Fit the scaler on the train data of the current patient
    dataset_train[['normal_samples']] = \
        min_max_scaler.fit_transform(dataset_train[['normal_samples']].values.reshape(-1, 1))

    # Transform the test data using the patient-specific scaler
    dataset_test[['normal_samples']] = \
        min_max_scaler.transform(dataset_test[['normal_samples']].values.reshape(-1, 1))
    # Normalizing per patient: normalize each patient's data to its own statistics
    for p in dataset_train['patient_id'].unique():
        # Get subset of the dataset for the current patient
        patient_data_train = dataset_train[dataset_train['patient_id'] == p]
        patient_data_test = dataset_test[dataset_test['patient_id'] == p]

        # Exclude 'normal_samples' column from normalization
        columns_to_normalize = patient_data_train.iloc[:, 12:].columns.difference(['normal_samples','label'])

        # Initialize a scaler for each patient
        scaler = StandardScaler()

        # Fit the scaler on the train data of the current patient
        dataset_train.loc[dataset_train['patient_id'] == p, columns_to_normalize] = \
            scaler.fit_transform(patient_data_train[columns_to_normalize])
        if p in dataset_test['patient_id'].unique():
          # Transform the test data using the patient-specific scaler
          transformed_data = scaler.transform(patient_data_test[columns_to_normalize])
          dataset_test.loc[dataset_test['patient_id'] == p, columns_to_normalize] = transformed_data


# Check if Nan exists
nan_positions = np.where(dataset_train.isna())
for row, col in zip(nan_positions[0], nan_positions[1]):
    print(f"Row: {row}, Column: {dataset_train.columns[col]}")
print(f'train nan: {len(nan_positions[0])}')

nan_positions = np.where(dataset_test.isna())
for row, col in zip(nan_positions[0], nan_positions[1]):
    print(f"Row: {row}, Column: {dataset_train.columns[col]}")
print(f'test nan: {len(nan_positions[0])}')


# Selecting columns except patient_id, date, start_time, end_time for learning process
#  Define inputs and label arrays
X_train, y_train = dataset_train.iloc[:, 3:].drop(columns=['label']), dataset_train['label'].astype(int)
X_test, y_test = dataset_test.iloc[:, 3:].drop(columns=['label']), dataset_test['label'].astype(int)

# For fair comparison with mlp that uses early stopping, we shring the number of training set for ML models
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)

X_columns = X_train.columns
n_features = X_train.shape[1]


## Train a model and get the results on various metrics
# Generate synthetic dataset

# Define and train the MLP model
if model_name == 'MLP':    

    # Define the objective function for Optuna
    def objective(trial):
        # Define hyperparameter search space
        #hidden_layer_1 = trial.suggest_int('hidden_layer_1', 32, 256)
        #hidden_layer_2 = trial.suggest_int('hidden_layer_2', 16, 128)
        alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-2)
        learning_rate_init = trial.suggest_loguniform('learning_rate_init', 1e-4, 1e-1)

        # Initialize the MLP model
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=1,  # We will control iterations manually
            warm_start=True,  # Enable continuation of training
            random_state=seed
        )

        # Early stopping parameters
        max_epochs = 300
        patience = 10
        best_val_score = -float('inf')
        epochs_without_improvement = 0

        for epoch in range(max_epochs):
            model.fit(X_train_sub, y_train_sub)  # Train for a single epoch
            val_predictions = model.predict(X_val)
            val_score = accuracy_score(y_val, val_predictions)

            if val_score > best_val_score:
                best_val_score = val_score
                epochs_without_improvement = 0
                # Optionally, save the best model parameters
                best_model = copy.deepcopy(model)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        return best_val_score  # Optuna aims to maximize this score

    # Run the optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)
    print("Best validation accuracy:", study.best_value)

    # Train the best model on the full training set
    best_params = study.best_params
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init'],
        max_iter=1,
        random_state=seed,
        warm_start=True  # ✅ Allows model to continue training across epochs
    )

    # Early stopping parameters
    max_epochs = 300
    patience = 10
    best_val_score = -float('inf')
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.fit(X_train_sub, y_train_sub)  # Train for a single epoch
        val_predictions = model.predict(X_val)
        val_score = accuracy_score(y_val, val_predictions)

        if val_score > best_val_score:
            best_val_score = val_score
            epochs_without_improvement = 0
            # Optionally, save the best model parameters
            best_model = copy.deepcopy(model)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break
    model = copy.deepcopy(best_model)


elif model_name == 'SVM':
    # Hyperparametertunning
    # Define the objective function for Optuna
    def objective(trial):
        # Define hyperparameter search space
        C = trial.suggest_loguniform('C', 1e-3, 1e3)  # Log-uniform sampling
        gamma = trial.suggest_loguniform('gamma', 1e-4, 1e1)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])

        # Create and evaluate model
        model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=seed)
        # Train the model
        model.fit(X_train_sub, y_train_sub)

        # Evaluate on the validation set
        val_predictions = model.predict(X_val)
        score = accuracy_score(y_val, val_predictions)
        
        return score  # Optuna tries to maximize this score

    # Run the optimization
    study = optuna.create_study(direction='maximize')  # Maximize accuracy
    study.optimize(objective, n_trials=20)  # Try 20 different hyperparameter sets

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)

    # Get the best hyperparameters from Optuna
    best_params = study.best_params
    print("Best parameters:", best_params)

    # Train the model using the best parameters
    model = SVC(**best_params, probability=True, random_state=seed)
    model.fit(X_train_sub, y_train_sub)

elif model_name == 'DecisionTree':
    # Define the objective function for Optuna
    def objective(trial):
        # Define hyperparameter search space
        max_depth = trial.suggest_int('max_depth', 3, 20)  # Depth of tree
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)  # Min samples to split
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)  # Min samples per leaf
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])  # Splitting method

        # Create and evaluate model
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=seed
        )
        
        # Train the model
        model.fit(X_train_sub, y_train_sub)

        # Evaluate on the validation set
        val_predictions = model.predict(X_val)
        score = accuracy_score(y_val, val_predictions)
        
        return score  # Optuna tries to maximize this score

    # Run the optimization
    study = optuna.create_study(direction='maximize')  # Maximize accuracy
    study.optimize(objective, n_trials=20)  # Try 20 different hyperparameter sets

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)

    # Get the best hyperparameters from Optuna
    best_params = study.best_params

    # Train the best model on the full training set
    model = DecisionTreeClassifier(**best_params, random_state=seed)
    model.fit(X_train_sub, y_train_sub)

else:
    raise ValueError("Unsupported model name")

# Predict and evaluate the model
y_pred = model.predict(X_test)
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)

# Calculate specificity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

# Calculate ROC AUC
roc_auc = roc_auc_score(y_test, y_pred)

# Calculate PR AUC
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred)
pr_auc = auc(recall_curve, precision_curve)

# Print the metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Sensitivity (Recall): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"PR AUC: {pr_auc:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

# Shaply kernel training and computing shap values
explainer = shap.KernelExplainer(model.predict, X_train_sub)
shap_values = explainer.shap_values(X_test)

# Create a single figure for visualizing feature importance
#plt.figure(figsize=(10, 6))
# Plot summary of feature importance
#shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
#plt.title('Feature Importance (Stacked)')
#plt.savefig(f'/content/drive/MyDrive/datasets/shap_{filename}.png', dpi=300, bbox_inches='tight')


# Calculate cumulative importance of features
shap_values_mean = np.mean(np.abs(shap_values), axis=0)  # mean absolute SHAP values
feature_importance = pd.DataFrame({'feature': range(X_train.shape[1]), 'importance': shap_values_mean})
feature_importance = feature_importance.sort_values(by='importance', ascending=False).reset_index(drop=True)

# Calculate cumulative importance
feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum() / feature_importance['importance'].sum()

# Display feature importance and cumulative importance
print(feature_importance)

# Identify negative impact features
negative_impact_features = [i for i in range(X_train.shape[1]) if (shap_values[:, i] < 0).mean() > 0.5]
print("\nFeatures with negative impact:", X_columns[negative_impact_features])

# Check correlation with target
correlation = pd.DataFrame(X_train).corrwith(pd.Series(y_train))
harmful_features = [feature for feature in negative_impact_features if correlation[feature] < 0]
print("\nHarmful features:", X_columns[harmful_features])
if if_global:
    norm_type = 'global'
else:
    norm_type = 'personalized'

shap_results = {'dataset':filename,'model':model_name,'normalization':norm_type,'accuracy':accuracy, 'f1':f1,'precision':precision,'recall':sensitivity, 'specificity':specificity,
                'pr_auc':pr_auc,'roc_auc':roc_auc,  'harmul_features':X_columns[harmful_features],
                'negative_impact_features':  X_columns[negative_impact_features]}

save_to_csv(shap_results,'/content/drive/MyDrive/datasets/results.csv')

## Remove the harmful features and retrain
if retrain:
    # Selecting columns except patient_id, date, start_time, end_time for learning process
    #  Define inputs and label arrays
    X_train, y_train = dataset_train.iloc[:, 3:].drop(columns=['label']), dataset_train['label'].astype(int)
    X_test, y_test = dataset_test.iloc[:, 3:].drop(columns=['label']), dataset_test['label'].astype(int)

    X_train = X_train.drop(columns= X_columns[harmful_features])
    X_test = X_test.drop(columns= X_columns[harmful_features])

    # For fair comparison with mlp that uses early stopping, we shring the number of training set for ML models
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)

    X_columns = X_train.columns
    n_features = X_train.shape[1]


    ## Train a model and get the results on various metrics
    # Generate synthetic dataset

    # Define and train the MLP model
    if model_name == 'MLP':    

        # Define the objective function for Optuna
        def objective(trial):
            # Define hyperparameter search space
            #hidden_layer_1 = trial.suggest_int('hidden_layer_1', 32, 256)
            #hidden_layer_2 = trial.suggest_int('hidden_layer_2', 16, 128)
            alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-2)
            learning_rate_init = trial.suggest_loguniform('learning_rate_init', 1e-4, 1e-1)

            # Initialize the MLP model
            model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=1,  # We will control iterations manually
                warm_start=True,  # Enable continuation of training
                random_state=seed
            )

            # Early stopping parameters
            max_epochs = 300
            patience = 10
            best_val_score = -float('inf')
            epochs_without_improvement = 0

            for epoch in range(max_epochs):
                model.fit(X_train_sub, y_train_sub)  # Train for a single epoch
                val_predictions = model.predict(X_val)
                val_score = accuracy_score(y_val, val_predictions)

                if val_score > best_val_score:
                    best_val_score = val_score
                    epochs_without_improvement = 0
                    # Optionally, save the best model parameters
                    best_model = copy.deepcopy(model)
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    break

            return best_val_score  # Optuna aims to maximize this score

        # Run the optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)

        # Print the best hyperparameters
        print("Best hyperparameters:", study.best_params)
        print("Best validation accuracy:", study.best_value)

        # Train the best model on the full training set
        best_params = study.best_params
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=best_params['alpha'],
            learning_rate_init=best_params['learning_rate_init'],
            max_iter=1,
            random_state=seed,
            warm_start=True  # ✅ Allows model to continue training across epochs
        )
        # Early stopping parameters
        max_epochs = 300
        patience = 10
        best_val_score = -float('inf')
        epochs_without_improvement = 0

        for epoch in range(max_epochs):
            model.fit(X_train_sub, y_train_sub)  # Train for a single epoch
            val_predictions = model.predict(X_val)
            val_score = accuracy_score(y_val, val_predictions)

            if val_score > best_val_score:
                best_val_score = val_score
                epochs_without_improvement = 0
                # Optionally, save the best model parameters
                best_model = copy.deepcopy(model)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break
        model = copy.deepcopy(best_model)
        


    elif model_name == 'SVM':
        # Hyperparametertunning
        # Define the objective function for Optuna
        def objective(trial):
            # Define hyperparameter search space
            C = trial.suggest_loguniform('C', 1e-3, 1e3)  # Log-uniform sampling
            gamma = trial.suggest_loguniform('gamma', 1e-4, 1e1)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])

            # Create and evaluate model
            model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=seed)
            # Train the model
            model.fit(X_train_sub, y_train_sub)

            # Evaluate on the validation set
            val_predictions = model.predict(X_val)
            score = accuracy_score(y_val, val_predictions)
            
            return score  # Optuna tries to maximize this score

        # Run the optimization
        study = optuna.create_study(direction='maximize')  # Maximize accuracy
        study.optimize(objective, n_trials=20)  # Try 20 different hyperparameter sets

        # Print the best hyperparameters
        print("Best hyperparameters:", study.best_params)
        print("Best accuracy:", study.best_value)

        # Get the best hyperparameters from Optuna
        best_params = study.best_params
        print("Best parameters:", best_params)

        # Train the model using the best parameters
        model = SVC(**best_params, probability=True, random_state=seed)
        model.fit(X_train_sub, y_train_sub)

    elif model_name == 'DecisionTree':
        # Define the objective function for Optuna
        def objective(trial):
            # Define hyperparameter search space
            max_depth = trial.suggest_int('max_depth', 3, 20)  # Depth of tree
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)  # Min samples to split
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)  # Min samples per leaf
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])  # Splitting method

            # Create and evaluate model
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=seed
            )
            
            # Train the model
            model.fit(X_train_sub, y_train_sub)

            # Evaluate on the validation set
            val_predictions = model.predict(X_val)
            score = accuracy_score(y_val, val_predictions)
            
            return score  # Optuna tries to maximize this score

        # Run the optimization
        study = optuna.create_study(direction='maximize')  # Maximize accuracy
        study.optimize(objective, n_trials=20)  # Try 20 different hyperparameter sets

        # Print the best hyperparameters
        print("Best hyperparameters:", study.best_params)
        print("Best accuracy:", study.best_value)

        # Get the best hyperparameters from Optuna
        best_params = study.best_params

        # Train the best model on the full training set
        model = DecisionTreeClassifier(**best_params, random_state=seed)
        model.fit(X_train_sub, y_train_sub)

    else:
        raise ValueError("Unsupported model name")

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)

    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred)

    # Calculate PR AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall_curve, precision_curve)

    # Print the metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"PR AUC: {pr_auc:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")

    # Shaply kernel training and computing shap values
    explainer = shap.KernelExplainer(model.predict, X_train_sub)
    shap_values = explainer.shap_values(X_test)
    fname = filename+'_post'

    # Create a single figure for visualizing feature importance
    #plt.figure(figsize=(10, 6))

    # Plot summary of feature importance
    #shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    #plt.title('Feature Importance (Stacked)')
    #plt.savefig(f'/content/drive/MyDrive/datasets/shap_{fname}.png', dpi=300, bbox_inches='tight')


    # Calculate cumulative importance of features
    shap_values_mean = np.mean(np.abs(shap_values), axis=0)  # mean absolute SHAP values
    feature_importance = pd.DataFrame({'feature': range(X_train.shape[1]), 'importance': shap_values_mean})
    feature_importance = feature_importance.sort_values(by='importance', ascending=False).reset_index(drop=True)

    # Calculate cumulative importance
    feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum() / feature_importance['importance'].sum()

    # Display feature importance and cumulative importance
    print(feature_importance)

    # Identify negative impact features
    negative_impact_features = [i for i in range(X_train.shape[1]) if (shap_values[:, i] < 0).mean() > 0.5]
    print("\nFeatures with negative impact:", X_columns[negative_impact_features])

    # Check correlation with target
    correlation = pd.DataFrame(X_train).corrwith(pd.Series(y_train))
    harmful_features = [feature for feature in negative_impact_features if correlation[feature] < 0]
    print("\nHarmful features:", X_columns[harmful_features])

    shap_results = {'dataset':fname,'model':model_name,'normalization':norm_type,'accuracy':accuracy, 'f1':f1,'precision':precision,'recall':sensitivity, 'specificity':specificity,
                    'pr_auc':pr_auc,'roc_auc':roc_auc,  'harmul_features':X_columns[harmful_features],
                    'negative_impact_features':  X_columns[negative_impact_features]}

    save_to_csv(shap_results,'/content/drive/MyDrive/datasets/results.csv')

        



