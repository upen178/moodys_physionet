# -*- coding: utf-8 -*-
"""moody3_stats_features.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eUL6Z33H12sNedHcyLR3Zfp0XYqntHK_
"""



import numpy as np
# import os
import scipy as sp
import pandas as pd
# from glob import glob
# from IPython.display import display
# import matplotlib.pyplot as plt
import math
from skimage.restoration import denoise_wavelet
from scipy.signal import savgol_filter
from scipy.signal import medfilt
# import seaborn as sns 
# import pywt
# from tqdm.notebook import tqdm 
import warnings
warnings.filterwarnings("ignore")
# import plotly.express as px
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE, Isomap
# from sklearn.cluster import DBSCAN, KMeans
# import plotly.graph_objs as go
# from plotly.offline import init_notebook_mode, iplot
# from plotly import tools
from scipy.stats import moment, skew, kurtosis, entropy
import sys
# sys.path.append("/content/pyeeg")
# from pyeeg import *

"""##### Functions"""

#band pass filter between 0.5 and 50 hz
from scipy.signal import butter, lfilter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def median(signal):# input: numpy array 1D (one column)
    array=np.array(signal)   
    #applying the median filter
    med_filtered=sp.signal.medfilt(array, kernel_size=3) # applying the median filter order3(kernel_size=3)
    return  med_filtered # return the med-filtered signal: numpy array 1D

#notch filter apllied at 50hz
def Implement_Notch_Filter(time, band, freq, ripple, order, filter_type, data):
    from scipy.signal import iirfilter
    fs   = 1/time
    nyq  = fs/2.0
    low  = freq - band/2.0
    high = freq + band/2.0
    low  = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop',
                     analog=False, ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def SignalProcessing(df):
    
    """
    This function impelments the signal processing pipeline through
    1- Median Filter 
    2- band pass filter 
    3- wavelet_denoise
    4- savgol_filter
    INPUT -------> the raw signals
    OUTPUT ------> A dictionary that contains the denoised signals 
    """

    filtered_df = pd.DataFrame()
    for col in df.columns:
        
            
        t_signal=df[col].values # copie the signal values in 1D numpy array
        med_filtred=median(t_signal) # apply 3rd order median filter and store the filtred signal in med_filtred
        fs = 100
        lowcut = 0.5
        highcut = 49
        band_pass=butter_bandpass_filter(med_filtred, lowcut, highcut, fs, order=5)
        #notch=Implement_Notch_Filter(0.02, 1, 50, 1, 2, 'butter',band_pass)
        wavelet_denoise=denoise_wavelet(band_pass,method='BayesShrink',mode='hard',wavelet='sym9',wavelet_levels=5,rescale_sigma=True)
        clean_signals=savgol_filter(wavelet_denoise, 1111, 3,mode='wrap')
        filtered_df[col] = clean_signals

    return filtered_df

def windowing(df, step_size, window_size):
    window_index_pairs =[]
    for i in range(0,len(df),step_size):
        z= i+window_size
        if z>len(df):
            break
        # print(i, z)
        window_index_pairs.extend([[i,z]])
    return window_index_pairs

def t_energy_axial(X) :
    energy_vector=(X**2).sum() # energy value of each df column
    return energy_vector # return energy vector energy_X,energy_Y,energy_Z

def stats_moment(array) :

    # default params moment
    moment_1 = moment(array, moment=1)
    # default moment = 2
    moment_2 = moment(array, moment=2)
    # default moment = 3
    moment_3 = moment(array, moment=3)
    # default moment = 4
    moment_4 = moment(array, moment=4)

    return moment_1, moment_2, moment_3, moment_4

def signal_amplitude(signal) :
    fft_spectrum = np.fft.rfft(signal)
    fft_spectrum_abs = np.abs(fft_spectrum)
    return np.sum(fft_spectrum_abs)

from scipy.stats import entropy
def stats_features(array) :

    # convert into pandas df
    df = pd.DataFrame(array)

    # extract featues
    stats_df = df.describe().T
    stats_df['skew'] = skew(df)
    stats_df['kurtosis'] = kurtosis(df)
    stats_df['energy'] = t_energy_axial(array)  
    stats_df['moment1'], stats_df['moment2'], stats_df['moment3'], stats_df['moment4'] = stats_moment(array)
    stats_df['amplitude'] = signal_amplitude(array)
    stats_df['entropy'] = entropy(array)


    # remove count column
    stats_df.drop(['count'], axis=1, inplace=True)

    return stats_df

def Segment_to_Features(array) :

    stats_df = stats_features(array)
    # rise_time_features = rise_decay_time(array)
    # peak_features = peaks_features(array)

    # feature_df = pd.concat([stats_df, rise_time_features, peak_features], axis=1)

    # return feature_df
    return stats_df

"""##### Data Generation"""

def statistical_features(df,Fs = 100):
    df.reset_index(drop= True, inplace= True)
    channel_cols =['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6',
       'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4',
       'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    patient_cols =['Patient','Age','Sex','ROSC','OHCA','VFib','TTM','Outcome','CPC']
    record_cols = ['Hour','Time','Quality','Record']
    df_group =df.groupby('Record')

    patient_feature = pd.DataFrame()
    for record in df['Record'].unique():
        df1 = df_group.get_group(record)
        df1.reset_index(drop=True, inplace= True)
        file_df = df1[channel_cols]

    # ## Multiplying with 10^6 to get significant numbers (Unit Conversion).Its assumed raw data in volts and we are converting to micro volts or Arbitrary Units(AU).
    # file_df[file_df.select_dtypes(include ='float64').columns] = file_df[file_df.select_dtypes(include ='float64').columns] *1e6
    
   
    

   
    

        # filter and preprocess signal
        filtered_df = SignalProcessing(file_df.select_dtypes(include= 'float64')) 

        # column wise feature calculation and concat with df
        df_features = pd.DataFrame()
        # generate window indcies
        # window_index_pairs = windowing(filtered_df,
        #                                step_size=10*Fs,
        #                                window_size= 30*Fs)
        for cols in filtered_df.columns:

            df_feat = pd.DataFrame()
            

            try :
                # extract segment
                X=filtered_df[cols].values
                # calculate stats features
                temp_df = Segment_to_Features(X)
                # concat data row wise
                df_feat = pd.concat([df_feat, temp_df], axis=0)

            except :
                pass

            # reset index
            df_feat.reset_index(drop=True, inplace=True)
            # print(df_feat.columns)
            # rename columns
            df_feat.columns = [cols+"_"+i for i in df_feat.columns]      
            # concat df column wise   
            df_features = pd.concat([df_features,df_feat], axis = 1)
        
        
        df_features[record_cols] =df1[record_cols].loc[0]
        patient_feature = pd.concat([patient_feature, df_features],axis =0)
    
    # patient_feature[patient_cols] = df[patient_cols].loc[0]
    try:
        patient_feature[patient_cols] = df[patient_cols].loc[0]
    except:
        # patient_cols1 =[colw for colw in patient_cols if colw not in ['Outcome','CPC']]
        # patient_feature[patient_cols1] = df[patient_cols1].loc[0]
        patient_feature.reset_index(drop=True,inplace=True)
        return patient_feature

    patient_feature.reset_index(drop=True,inplace=True)

    return patient_feature
        


    # df_features['Subject'] = file.split('/')[-1][:-4]   
    # for col1 in ['Time']:
    #     df_features[col1] = file_df[col1][np.array(window_index_pairs)[:,1].flatten()-1].values

    
    # df_new_name = file.split("/")[-1]
    
    # df_features.to_csv(f'/content/drive/MyDrive/EEG_DatasetDRL/stats_features_wmed/stats_{df_new_name}', index=False)


    # del file_df
    # break