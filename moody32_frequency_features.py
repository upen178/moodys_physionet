# -*- coding: utf-8 -*-
"""moody32_frequency_features.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LGxY-FwJpAmfCOn2XnqgcMvwnP0PY-Oj

Objective
1. Importand electrode selection, File Proprocessing and unit conversion
2. windowing (window_size = 30Secs, step_size =10Secs)
3. Frequency domain feature generation in freq bands [0.5,4,8,12,30,50]
 3.1 delta band 0.5- 4 Hz,theta 4-8 Hz, alfa band 8-12 Hz, beta 12-30Hz, gama band 30-50 Hz.
4. in feature file Total 12 features for each channel(17*12= 204) and two additional columns subject and Time( total =206)
"""

import pandas as pd
from scipy import signal
import numpy as np
from scipy.signal import welch
from scipy.integrate import simps
# import entropy
# import entropy as ent

import pandas as pd
import os
import numpy as np
# import plotly.graph_objs as go
# from plotly.offline import init_notebook_mode, iplot
# from plotly import tools

# import mne
# Importing numpy 
import numpy as np
# Importing Scipy 
import scipy as sp
# Importing Pandas Library 
import pandas as pd
# import glob function to scrap files path
from glob import glob
# import display() for better visualitions of DataFrames and arrays
# from IPython.display import display
# import pyplot for plotting
# import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from scipy.signal import savgol_filter
from scipy.signal import medfilt
# import seaborn as sns
# import pywt
# from tqdm import tqdm


"""### Appending Paths

### Step1. Signal Preprocessing

#### Band pass filtering
 ###### https://www.kaggle.com/code/sam1o1/eeg-signal-processing/notebook
 ###### Channel Selection https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9848795
 #### https://braininformatics.springeropen.com/articles/10.1186/s40708-022-00159-3
"""

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

"""#### Median Filtering """

from scipy.signal import medfilt # import the median filter function
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

# def SignalProcessing(raw_dic):
def SignalProcessing(df):
    
    """
    This function impelments the signal processing pipeline through
    1- Median Filter 
    2-band pass filter 
    3-wavelet_denoise
    4-savgol_filter
    INPUT -------> the raw signals
    OUTPUT ------> A dictionary that contains the denoised signals 
    """
    # time_sig_dic={} # An empty dictionary will contains dataframes of all time domain signals
    filtered_df = pd.DataFrame()
    # raw_dic_keys=sorted(raw_dic.keys()) # sorting dataframes' keys
    # for key in raw_dic_keys:
    # for key in raw_dic_keys:
    for col in df.columns:
        
        # raw_df=raw_dic[key]
        # time_sig_df=pd.DataFrame()
        # for column in raw_df.columns:
            
        t_signal=df[col].values # copie the signal values in 1D numpy array
        med_filtred=median(t_signal) # apply 3rd order median filter and store the filtred signal in med_filtred
        fs = 100
        lowcut = 0.5 ## lower cut off frequency
        highcut = 49 ## higher cut off frequency
        band_pass=butter_bandpass_filter(med_filtred, lowcut, highcut, fs, order=5)
        #notch=Implement_Notch_Filter(0.02, 1, 50, 1, 2, 'butter',band_pass)
        wavelet_denoise=denoise_wavelet(band_pass,method='BayesShrink',mode='hard',wavelet='sym9',wavelet_levels=5,rescale_sigma=True)
        clean_signals=savgol_filter(wavelet_denoise, 1111, 3,mode='wrap')
        filtered_df[col] = clean_signals
        # time_sig_df[column]=clean_signals
        # time_sig_dic[key]=time_sig_df
    # return time_sig_dic
    return filtered_df

"""### Step2.  Feature Generation

#### Step2.1 WIndowing
"""

def windowing(df, step_size, window_size):
    window_index_pairs =[]
    for i in range(0,len(df),step_size):
        z= i+window_size
        if z>len(df):
            break
        # print(i, z)
        window_index_pairs.extend([[i,z]])
    return window_index_pairs

"""### Freq_domain features

#### https://raphaelvallat.com/bandpower.html
#### https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3347767/
"""

def feature_bandpower(data,col_name, sf =100, bands=[0.5,4,8,12,30,50]):
    """Compute the average power of the signal x in a specific frequency band.



    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    sf : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]
    
    #Total Power
    total_power = simps(psd, dx=freq_res)

    dict1 ={}
    dict1[col_name +'_totpowr'] =total_power
    # from mne.time_frequency import psd_array_multitaper
    power =[]
    power_ratio=[]
    for i in range(len(bands)-1): ### looping over frequency bands
        band = bands[i:i+2]

        band = np.asarray(band)
        low, high = band

        

        # Find index of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs < high)

        # Integral approximation of the spectrum using parabola (Simpson's rule)
        bp = simps(psd[idx_band], dx=freq_res)
        bp_ratio = bp/total_power
        power.extend([bp])
        power_ratio.extend([bp_ratio])


        



    #### appending power

    dict1[col_name +'_deltapowr'] =power[0]
    dict1[col_name +'_thetapowr'] =power[1]
    dict1[col_name +'_alfapowr'] =power[2]
    dict1[col_name +'_betapowr'] =power[3]
    dict1[col_name +'_gamapowr'] =power[4]

    #### appending power ratio
    
    dict1[col_name +'_deltapowr_r'] =power_ratio[0]
    dict1[col_name +'_thetapowr_r'] =power_ratio[1]
    dict1[col_name +'_alfapowr_r'] =power_ratio[2]
    dict1[col_name +'_betapowr_r'] =power_ratio[3]
    dict1[col_name +'_gamapowr_r'] =power_ratio[4]

    ## Entropy source code copied from git entropy clone
    psd_norm = psd / psd.sum(axis=-1, keepdims=True)
    se = -(psd_norm * np.log2(psd_norm)).sum(axis=-1)
    dict1[col_name +'_spectral_ent'] =se

    ### power ratio with each other

    # dict1[col_name +'_delta/theta'] =power[0]/power[4]
    # dict1[col_name +'_theta/alfa'] =power[1]
    # dict1[col_name +'_alfa/beta'] =power[2]
    # dict1[col_name +'_beta/gama'] =power[3]/power[4]
    # dict1[col_name +'_gamapowr'] =
    

    return dict1

"""#### genrating feature dataframe"""

def frequency_features(df,Fs = 100):
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
                temp_df = feature_bandpower(X,cols)
                temp_df= pd.DataFrame(temp_df,index=[0])
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
    
    try:
        patient_feature[patient_cols] = df[patient_cols].loc[0]
    except:
        # patient_cols1 =[colw for colw in patient_cols if colw not in ['Outcome','CPC']]
        # patient_feature[patient_cols1] = df[patient_cols1].loc[0]
        patient_feature.reset_index(drop=True,inplace=True)
        return patient_feature
    patient_feature.reset_index(drop=True,inplace=True)
    return patient_feature