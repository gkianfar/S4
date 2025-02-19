import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
import matplotlib.patches as mpatches
from collections.abc import Iterable
from sklearn.preprocessing import StandardScaler
import os
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings('ignore')


# Configuration
if_global = True
seed = 2025 #  Set seed for repeatable train test val split
ratios = [0.9,0.1] # [train, test]
if sum(ratios) != 1:
    raise Exception('Sum of ratios should be 1')


filename = ''
dataset_path = f'/content/drive/MyDrive/datasets/dataset_{filename}.csv'
dataset = pd.read_csv(dataset_path)

# Encode nonnumeric features and correcting data types

# Modify dtype
dataset_train['normal_samples'] = dataset_train['normal_samples'].astype(float)
dataset_test['normal_samples'] = dataset_test['normal_samples'].astype(float)

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
  dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
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

    # Store global statistics
    global_stats['mean'] = scaler.mean_
    global_stats['std'] = scaler.scale_
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
          dataset_test.loc[dataset_test['patient_id'] == p, columns_to_normalize] = \
              scaler.transform(patient_data_test[columns_to_normalize])



