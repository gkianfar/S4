import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from matplotlib import cm
import matplotlib.patches as mpatches
from collections.abc import Iterable
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

## Set colour palette
ibm_colorblind = ['#648FFF', '#FE6100', '#DC267F', '#785EF0', '#FFB000','#48A9A6']
sns.set_palette(ibm_colorblind)


def correct_col_type(df,col):
    raw_type = str(type(df[col].dtype)).split('.')[-1].split('\'')[0].lower()
    #print(col,raw_type)
    if 'object' in raw_type:
        if 'date' in col or 'timestamp' in col or 'datetime' in col:
            return pd.to_datetime(df[col])
        else:
            return df[col].astype('category')
    else:
        return df[col]


def gen_date_col(df, tcol):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df[tcol].dt.date
    return df



def gen_date_col(df, tcol):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df[tcol].dt.date
    return df


def load_datasets(DPATH):

    ### Labels
    f = 'Labels.csv'
    fpth = os.path.join(DPATH,f)
    labels_df = pd.read_csv(fpth)
    for col in labels_df.columns:
        labels_df[col] = correct_col_type(labels_df,col)
    if 'date' in labels_df.columns:
        labels_df = labels_df.rename(columns={'date':'timestamp'})
        labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'])
        labels_df['date'] = labels_df['timestamp'].dt.date
        labels_df['time'] = labels_df['timestamp'].dt.time

    if 'type' in labels_df.columns:
        labels_df = labels_df.rename(columns={'type':'label'})

    ### Activity
    f = 'Activity.csv'
    fpth = os.path.join(DPATH,f)
    df = pd.read_csv(fpth)
    for col in df.columns:
        df[col] = correct_col_type(df,col)
    if 'date' in df.columns:
        activity_df = df.rename(columns={'date':'timestamp'})
        activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp'])
        activity_df['date'] = activity_df['timestamp'].dt.date
        activity_df['time'] = activity_df['timestamp'].dt.time


    ### Physiology
    f = 'Physiology.csv'
    fpth = os.path.join(DPATH,f)
    df = pd.read_csv(fpth)
    for col in df.columns:
        df[col] = correct_col_type(df,col)
    if 'date' in df.columns:
        physiology_df = df.rename(columns={'date':'timestamp'})
        physiology_df['timestamp'] = pd.to_datetime(physiology_df['timestamp'])
        physiology_df['date'] = physiology_df['timestamp'].dt.date
        physiology_df['time'] = physiology_df['timestamp'].dt.time

    # Sleep
    f = 'Sleep.csv'
    fpth = os.path.join(DPATH,f)
    df = pd.read_csv(fpth)
    for col in df.columns:
        df[col] = correct_col_type(df,col)
    if 'date' in df.columns:
        sleep_df = df.rename(columns={'date':'timestamp'})
        sleep_df['timestamp'] = pd.to_datetime(sleep_df['timestamp'])
        sleep_df['date'] = sleep_df['timestamp'].dt.date
        sleep_df['time'] = sleep_df['timestamp'].dt.time

    # Demographics dataset
    f = 'Demographics.csv'
    fpth = os.path.join(DPATH,f)
    demographics_df = pd.read_csv(fpth)
    for col in demographics_df.columns:
        demographics_df[col] = correct_col_type(demographics_df,col)

    return activity_df, physiology_df, sleep_df, labels_df, demographics_df



def get_transition_matrix(activity_df, make_reciprocal=False):
    patients = activity_df['patient_id'].unique()
    states = activity_df['location_name'].unique().tolist()
    n_state = len(states)
    transition_matrix = {}

    state_map = dict(zip(states, range(n_state)))
    state_map_reverse = {v:k for k, v in state_map.items()}

    activity_df['location_name'] = activity_df['location_name'].copy().map(state_map)



    for p in patients:
        transition_matrix[p] = np.zeros((n_state, n_state))
        days = activity_df['date'][activity_df['patient_id'] == p].unique()
        days = np.sort(days)
        flag = True

        for i, d in enumerate(days):
            df = activity_df[(activity_df['patient_id'] == p) & (activity_df['date'] == d)]
            sequence = df['location_name'].tolist()
            timestamp = df['timestamp'].tolist()
            if not sequence:
                continue

            if flag:
                init_state = sequence[0]
                init_time = timestamp[0]

            for s, t in zip(sequence[1:], timestamp[1:]):

                transition_matrix[p][init_state][s] += 1
                init_state = s
                init_time = t


            if i != len(days) - 1:
                if (days[i+1] - d).days == 1:
                    flag = False
                else:
                    flag = True

    for p in patients:
        for i in range(n_state):
            row_sum = np.sum(transition_matrix[p][i, :])
            if row_sum != 0:
                transition_matrix[p][i, :] /= row_sum
    # Make the matrices reciprocal
    if make_reciprocal:
      for p in patients:
          transition_matrix[p] = np.minimum(transition_matrix[p], transition_matrix[p].T)
          transition_matrix[p] = transition_matrix[p] / np.sum(transition_matrix[p], axis=1, keepdims=True)

    return transition_matrix, state_map_reverse

def get_impossible_count(activity_df,transition_matrix,threshold,occurence_threshold):
  patients = activity_df['patient_id'].unique()

  impossible_activity_count = {}

  for p in patients:

      impossible_activity_count[p] = []

      patient_matrix = transition_matrix[p]  # Extract patient's transition matrix
      indices = np.where((patient_matrix < threshold))  # Find indices

      patient_impossible_transitions = list(zip(*indices))
      days = activity_df['date'][activity_df['patient_id'] == p].unique()
      days = np.sort(days)

      for i, d in enumerate(days):
          counter = 0
          df = activity_df[(activity_df['patient_id'] == p) & (activity_df['date'] == d)]
          sequence = df['location_name'].tolist()
          timestamp = df['timestamp'].tolist()
          if not sequence:
              continue

          for s, t in zip(sequence[1:], timestamp[1:]):

              init_state = s
              init_time = t
              if (init_state,s) in patient_impossible_transitions:
                counter+=1
          if counter>occurence_threshold:
            impossible_activity_count[p].append((d,counter))
      if not impossible_activity_count[p]:
        del impossible_activity_count[p]
  return impossible_activity_count