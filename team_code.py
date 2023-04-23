#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib 
import pandas as pd 
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier 
import wfdb 
from moody3_stats_features import statistical_features 
from moody32_frequency_features import frequency_features 

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose): 
    data_folders = find_data_folders(data_folder)   

    data_folders = data_folders[:2] 
    # print(data_folders)  

    features = feat_gen(data_folders, data_folder)
    
    features = drived_features(features)  

    # print(features.columns)  

    col_list = None 
    with open('col.txt') as f:
        col_list = f.read().split('\n') 

    # print(contents) 

    # print(type(contents))

    x_train = features[col_list] 
    y_train = features['CPC']  

    
    # generate smote data
    smote_clf = SMOTE(n_jobs=-1)

    try: 
        sm_xtrain, sm_ytrain = smote_clf.fit_resample(x_train, y_train)

        # # extract major class value
        major_label = y_train.value_counts().index[0]

        # remove major label's data from smote data
        sm_xtrain = sm_xtrain[sm_ytrain != major_label] 
        sm_ytrain = sm_ytrain[sm_ytrain != major_label]

        x_train = pd.concat([x_train, sm_xtrain], axis=0)
        y_train = pd.concat([y_train, sm_ytrain], axis=0)
    except:
        pass 

    rf_ts = RandomForestRegressor(max_depth=14, max_features='log2', min_samples_leaf=4,
                       min_samples_split=6, n_estimators=150)

    rf_ts.fit(x_train, y_train)   
    

    # classification  

    x_train1 = features[col_list] 
    y_train1 = features['Outcome']   

    
    # # generate smote data
    # smote_clf = SMOTE(n_jobs=-1)
    # sm_xtrain1, sm_ytrain1 = smote_clf.fit_resample(x_train1, y_train1)

    # # # extract major class value
    # major_label = y_train1.value_counts().index[0]

    # # remove major label's data from smote data
    # sm_xtrain1 = sm_xtrain1[sm_ytrain1 != major_label] 
    # sm_ytrain1 = sm_ytrain1[sm_ytrain1 != major_label] 
    
    lgb_ts = LGBMClassifier(boosting='gbdt', learning_rate=0.01, max_depth=10,
                            num_iterations=5000, num_leaves=35)

    lgb_ts.fit(x_train1, y_train1)   

    save_challenge_model(model_folder, None, lgb_ts, rf_ts) 


def drived_features(df):
    ## Extracting Channel_list
    channel_list=[]
    for col in df.columns:
        if col.endswith('_alfapowr'):
            channel_list.extend([col.rstrip('_alfapowr')])

    # channel_list
    
    for channel in channel_list:
        alfa = channel+'_alfapowr'
        beta = channel+'_betapowr'
        gama = channel+'_gamapowr'
        theta = channel+'_thetapowr'
        df[channel+'_alfa+beta+theta-2*gama'] = df[alfa]+df[beta]+df[theta]-(2*df[gama])
        df[channel+'_alfa+beta+theta-gama'] = df[alfa]+df[beta]+df[theta]-df[gama]
        df[channel+'_alfa+beta+theta'] = df[alfa]+df[beta]+df[theta] 

    
    return df

    

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename) 


# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    # imputer = models['imputer']
    outcome_model = models['outcome_model'] 
    cpc_model = models['cpc_model']

    # Load data.
    # patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id) 

    # Extract features.
    # features = get_features(patient_metadata, recording_metadata, recording_data) 

    # cwd = os.getcwd()

    # data_folders = os.path.join(cwd, data_folder, patient_id)

    # data_folders = find_data_folders(data_folder)   

    features = feat_gen_test(data_folder,patient_id) 
    
    features = drived_features(features)

    col_list = None 
    with open('col.txt') as f:
        col_list = f.read().split('\n') 

    # print(contents) 

    # print(type(contents))

    x_test = features[col_list]  

    # print(x_test.head(10)) 
    # y_test = features['CPC']  
    

    # Apply models to features.  
    # outcome = outcome_model.predict(x_test)[0]  
    outcome = outcome_model.predict(x_test)[-1] 

    if outcome == 'Good':
        outcome = 0
    else: 
        outcome = 1 


    print(outcome)
    # outcome_probability = outcome_model.predict_proba(x_test)[0, 1] 
    outcome_probability = outcome_model.predict_proba(x_test)[-1,1]

    print(outcome_probability) 
    # cpc = cpc_model.predict(x_test)[0]  
    cpc = cpc_model.predict(x_test)[-1]

    print(cpc) 

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return str(outcome), outcome_probability, float(cpc) 

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data):
    # Extract features from the patient metadata.
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables
    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    # Combine the patient features.
    patient_features = np.array([age, female, male, other, rosc, ohca, vfib, ttm])

    # Extract features from the recording data and metadata.
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    num_channels = len(channels)
    num_recordings = len(recording_data)

    # Compute mean and standard deviation for each channel for each recording.
    available_signal_data = list()
    for i in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
            available_signal_data.append(signal_data)

    if len(available_signal_data) > 0:
        available_signal_data = np.hstack(available_signal_data)
        signal_mean = np.nanmean(available_signal_data, axis=1)
        signal_std  = np.nanstd(available_signal_data, axis=1)
    else:
        signal_mean = float('nan') * np.ones(num_channels)
        signal_std  = float('nan') * np.ones(num_channels)

    # Compute the power spectral density for the delta, theta, alpha, and beta frequency bands for each channel of the most
    # recent recording.
    index = None
    for i in reversed(range(num_recordings)):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            index = i
            break

    if index is not None:
        signal_data, sampling_frequency, signal_channels = recording_data[index]
        signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.

        delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)

        quality_score = get_quality_scores(recording_metadata)[index]
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)
        quality_score = float('nan')

    recording_features = np.hstack((signal_mean, signal_std, delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean, quality_score))

    # Combine the features from the patient metadata and the recording data and metadata.
    features = np.hstack((patient_features, recording_features))

    return features



def feat_gen(data_folders, path_root): 

    for patient_id in data_folders: 
        print("patient id : ", patient_id)  
        df_all_comb = pd.DataFrame()  
        comb_df = pd.DataFrame() 
        # print('This is for',patient_id) 
        # Define file location.
        patient_metadata_file = os.path.join(path_root, patient_id, patient_id + '.txt')
        recording_metadata_file = os.path.join(path_root, patient_id, patient_id + '.tsv')
        df_rec_met = pd.read_csv(recording_metadata_file, sep='\t')
        df_rec_met.dropna(inplace= True)
        df_rec_met.reset_index(drop= True, inplace= True)
        df_pat_met = patient_metadata(patient_metadata_file)

        for ind,patient_id_sub in enumerate(df_rec_met['Record']):
            # print(patient_id_sub) 
            record1 =wfdb.rdrecord(os.path.join(path_root,patient_id, patient_id_sub))
            df= pd.DataFrame(record1.p_signal)
            df.columns = record1.sig_name
            for col in df_rec_met.columns:
                df[col] = df_rec_met[col][ind]
            comb_df = pd.concat([comb_df, df])
        
        for col1 in df_pat_met.columns:
            comb_df[col1] = df_pat_met[col1][0]

                        
        
        # comb_df.to_csv(os.path.join(com_path, patient_id+'.csv'))
        comb_df.reset_index(drop=True,inplace=True)
        statistical_feature  = statistical_features(comb_df)
        freq_feature  = frequency_features(comb_df)

        df_all =freq_feature.merge(statistical_feature, on=['Hour', 'Time', 'Quality', 'Record', 'Patient', 'Age', 'Sex', 'ROSC',
        'OHCA', 'VFib', 'TTM', 'Outcome', 'CPC'] )

        # path_write = os.path.join(path_w,patient_id+'_stat.csv')
        # statistical_feature.to_csv(path_write, index= False)

        print("done")
        # output.clear() 

        df_all_comb = pd.concat([df_all_comb, df_all], axis=0)    


    return df_all_comb



def feat_gen_test(data_folder, patient_id): 

    # for patient_id in data_folders: 
    print("patient id : ", patient_id)  
    # df_all_comb = pd.DataFrame()  
    comb_df = pd.DataFrame() 
    # print('This is for',patient_id) 
    # Define file location.
    # patient_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.txt')
    recording_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.tsv')
    df_rec_met = pd.read_csv(recording_metadata_file, sep='\t')
    df_rec_met.dropna(inplace= True)
    df_rec_met.reset_index(drop= True, inplace= True)
    # df_pat_met = patient_metadata(patient_metadata_file)

    for ind,patient_id_sub in enumerate(df_rec_met['Record']):
        # print(patient_id_sub) 
        record1 =wfdb.rdrecord(os.path.join(data_folder,patient_id, patient_id_sub))
        df= pd.DataFrame(record1.p_signal)
        df.columns = record1.sig_name
        for col in df_rec_met.columns:
            df[col] = df_rec_met[col][ind]
        comb_df = pd.concat([comb_df, df])
    
    col_list1 = None 
    with open('chennel_list.txt') as f:
        col_list1 = f.read().split('\n') 


    # for col1 in df_pat_met.columns:
    #     comb_df[col1] = df_pat_met[col1][0]
    # for col1 in col_list1:
        # comb_df[col1] = df_pat_met[col1][0]

                    
    
    # comb_df.to_csv(os.path.join(com_path, patient_id+'.csv'))
    comb_df.reset_index(drop=True,inplace=True)
    statistical_feature  = statistical_features(comb_df)
    freq_feature  = frequency_features(comb_df)

    try:
        df_all =freq_feature.merge(statistical_feature, on=['Hour', 'Time', 'Quality', 'Record', 'Patient', 'Age', 'Sex', 'ROSC',
        'OHCA', 'VFib', 'TTM', 'Outcome', 'CPC'] )
    except:
        df_all =freq_feature.merge(statistical_feature, on=['Hour', 'Time', 'Quality', 'Record'] )

    # path_write = os.path.join(path_w,patient_id+'_stat.csv')
    # statistical_feature.to_csv(path_write, index= False)

    print("done")
    # output.clear() 

    # df_all_comb = pd.concat([df_all_comb, df_all], axis=0)    


    return df_all


def patient_metadata(patient_metadata_file):
    df_txt = pd.read_csv(patient_metadata_file, sep=':', header = None)
    df_txt[0] = df_txt[0].str.strip() 
    df_txt[1] = df_txt[1].str.strip() 
    df_txt = df_txt.T
    df_txt.columns = df_txt.loc[0]
    df_txt = df_txt[1:]
    df_txt.reset_index(drop= True, inplace = True)
    return df_txt

 