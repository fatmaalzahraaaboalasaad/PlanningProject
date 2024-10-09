#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:57:15 2024

@author: fatma
"""


 #%% 
import os
import h5py
from math import isnan
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.ndimage import gaussian_filter as gaussian_filter
from statsmodels.formula.api import ols
from scipy.stats import probplot
import os
import glob
from scipy.stats import spearmanr
import scipy.stats
import scikit_posthocs as sp
from scipy.stats import shapiro
from statsmodels.stats.anova import AnovaRM
import pingouin as pg
from scipy.stats import ttest_rel
from scipy import stats
from numpy import random
from scipy.stats import ttest_ind
from mlxtend.evaluate import permutation_test
from scipy.stats import kstest
from matplotlib.ticker import MaxNLocator
import matplotlib 
plt.rcParams.update({'font.size': 15})
from scipy.stats import probplot
from statsmodels.stats.diagnostic import het_white
import seaborn as sns
from statannot import add_stat_annotation
from statannotations.Annotator import Annotator
import os
import h5py
from math import isnan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
from scipy.ndimage import gaussian_filter as gauss
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy.stats as sp
from textwrap import wrap
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from statsmodels.formula.api import ols
import seaborn as sns
sns.set_theme()
import math
import matplotlib.path as mpath
import matplotlib.patches as mpatches

 #%% 
# set up of dataframes where the csv information of all participants will be stored 
global results
global results_0
global results_1
global results_2
global all_subj
global all_subj_0
global all_subj_1
global all_subj_2

participant=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
#participant=[1]


# dataframes of results: one row per participant which combines the data of all of trials form this subject
# numbers specific time frame of the decison phase: none = complete decison phase; 0 = stimulus onset until first choice; 1 first choice to second choice; 2 second choice until third choice

results = pd.DataFrame(columns=['participant','validgaze_percent'])
results['participant'] = participant

results_0 = pd.DataFrame(columns=['participant','validgaze_percent'])
results_0['participant'] = participant

results_1 = pd.DataFrame(columns=['participant','validgaze_percent'])
results_1['participant'] = participant

results_2 = pd.DataFrame(columns=['participant','validgaze_percent'])
results_2['participant'] = participant

# Dataframes of all trials and all participants
# numbers specific time frame of the decison phase: none = complete decison phase; 0 = stimulus onset until first choice; 1 first choice to second choice; 2 second choice until third choice
all_subj = pd.DataFrame(columns=['trial.thisN', 'participant', 'Condition', 'N_Scenario', 'A1pos',
       'A2pos', 'A3pos', 'B1pos', 'B2pos', 'B3pos', 'C1pos', 'C2pos', 'C3pos',
       'Imagesdecision_started', 'decisiontimeA', 'decisiontimeB',
       'decisiontimeC', 'clicked_optionA', 'clicked_optionB',
       'clicked_optionC', 'slider_confidence_response', 'decisiontimetotal',
       'decisiontime1', 'decisiontime2', 'Imagesdecision_start',
       'Imagesdecision_stopped', 'et_decision_start', 'et_decision_end',
       'valid_datapoints', 'n_3_1', 'n_3_2', 'n_3_3', 'n_3_4', 'n_3_5',
       'n_3_6', 'n_3_7', 'n_3_8', 'n_3_9', 't_A1', 't_A2', 't_A3', 't_B1',
       't_B2', 't_B3', 't_C1', 't_C2', 't_C3', 't_cat_total', 'f_categoryA',
       'f_categoryB', 'f_categoryC', 'first_look', 'first_switch', 'WC1',
       'WC2', 'WC3', 'B', 'W', 'W_B_total', 'decision1_category',
       'decision2_category', 'decision3_category', 'category1_imp',
       'category2_imp', 'category3_imp', 'category1_diff', 'category2_diff',
       'category3_diff', 'f_category1', 'f_category2', 'f_category3',
       'decision1_1category', 'decision1_2category', 'decision1_3category',
       'decision2_1category', 'decision2_2category', 'decision2_3category',
       'decision3_1category', 'decision3_2category', 'decision3_3category'])


all_subj_0 = pd.DataFrame(columns=['trial.thisN', 'participant', 'Condition', 'N_Scenario', 'A1pos',
       'A2pos', 'A3pos', 'B1pos', 'B2pos', 'B3pos', 'C1pos', 'C2pos', 'C3pos',
       'Imagesdecision_started', 'decisiontimeA', 'decisiontimeB',
       'decisiontimeC', 'clicked_optionA', 'clicked_optionB',
       'clicked_optionC', 'slider_confidence_response', 'decisiontimetotal',
       'decisiontime1', 'decisiontime2', 'Imagesdecision_start',
       'Imagesdecision_stopped', 'et_decision_start', 'et_decision_end',
       'valid_datapoints', 'n_3_1', 'n_3_2', 'n_3_3', 'n_3_4', 'n_3_5',
       'n_3_6', 'n_3_7', 'n_3_8', 'n_3_9', 't_A1', 't_A2', 't_A3', 't_B1',
       't_B2', 't_B3', 't_C1', 't_C2', 't_C3', 't_cat_total', 'f_categoryA',
       'f_categoryB', 'f_categoryC', 'first_look', 'first_switch', 'WC1',
       'WC2', 'WC3', 'B', 'W', 'W_B_total', 'decision1_category',
       'decision2_category', 'decision3_category', 'category1_imp',
       'category2_imp', 'category3_imp', 'category1_diff', 'category2_diff',
       'category3_diff', 'f_category1', 'f_category2', 'f_category3',
       'decision1_1category', 'decision1_2category', 'decision1_3category',
       'decision2_1category', 'decision2_2category', 'decision2_3category',
       'decision3_1category', 'decision3_2category', 'decision3_3category'])


all_subj_1 = pd.DataFrame(columns=['trial.thisN', 'participant', 'Condition', 'N_Scenario', 'A1pos',
       'A2pos', 'A3pos', 'B1pos', 'B2pos', 'B3pos', 'C1pos', 'C2pos', 'C3pos',
       'Imagesdecision_started', 'decisiontimeA', 'decisiontimeB',
       'decisiontimeC', 'clicked_optionA', 'clicked_optionB',
       'clicked_optionC', 'slider_confidence_response', 'decisiontimetotal',
       'decisiontime1', 'decisiontime2', 'Imagesdecision_start',
       'Imagesdecision_stopped', 'et_decision_start', 'et_decision_end',
       'valid_datapoints', 'n_3_1', 'n_3_2', 'n_3_3', 'n_3_4', 'n_3_5',
       'n_3_6', 'n_3_7', 'n_3_8', 'n_3_9', 't_A1', 't_A2', 't_A3', 't_B1',
       't_B2', 't_B3', 't_C1', 't_C2', 't_C3', 't_cat_total', 'f_categoryA',
       'f_categoryB', 'f_categoryC', 'first_look', 'first_switch', 'WC1',
       'WC2', 'WC3', 'B', 'W', 'W_B_total', 'decision1_category',
       'decision2_category', 'decision3_category', 'category1_imp',
       'category2_imp', 'category3_imp', 'category1_diff', 'category2_diff',
       'category3_diff', 'f_category1', 'f_category2', 'f_category3',
       'decision1_1category', 'decision1_2category', 'decision1_3category',
       'decision2_1category', 'decision2_2category', 'decision2_3category',
       'decision3_1category', 'decision3_2category', 'decision3_3category'])

all_subj_2 = pd.DataFrame(columns=['trial.thisN', 'participant', 'Condition', 'N_Scenario', 'A1pos',
       'A2pos', 'A3pos', 'B1pos', 'B2pos', 'B3pos', 'C1pos', 'C2pos', 'C3pos',
       'Imagesdecision_started', 'decisiontimeA', 'decisiontimeB',
       'decisiontimeC', 'clicked_optionA', 'clicked_optionB',
       'clicked_optionC', 'slider_confidence_response', 'decisiontimetotal',
       'decisiontime1', 'decisiontime2', 'Imagesdecision_start',
       'Imagesdecision_stopped', 'et_decision_start', 'et_decision_end',
       'valid_datapoints', 'n_3_1', 'n_3_2', 'n_3_3', 'n_3_4', 'n_3_5',
       'n_3_6', 'n_3_7', 'n_3_8', 'n_3_9', 't_A1', 't_A2', 't_A3', 't_B1',
       't_B2', 't_B3', 't_C1', 't_C2', 't_C3', 't_cat_total', 'f_categoryA',
       'f_categoryB', 'f_categoryC', 'first_look', 'first_switch', 'WC1',
       'WC2', 'WC3', 'B', 'W', 'W_B_total', 'decision1_category',
       'decision2_category', 'decision3_category', 'category1_imp',
       'category2_imp', 'category3_imp', 'category1_diff', 'category2_diff',
       'category3_diff', 'f_category1', 'f_category2', 'f_category3',
       'decision1_1category', 'decision1_2category', 'decision1_3category',
       'decision2_1category', 'decision2_2category', 'decision2_3category',
       'decision3_1category', 'decision3_2category', 'decision3_3category'])



 #%% 

# Define Functions to transform the hdf5 data (eyetracking) to csv 

# function to find the closest time correspondence between the eyetracking recording and the start of images in each trial
def closest(lst, K):
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return lst[idx]


def get_planning_data(fileDir,subj):
    global trials
    global etData 
    files_subject = [_ for _ in os.listdir(fileDir) if (_.endswith(r".hdf5") or _.endswith(r".csv")) and "P"+str(subj) in _]
    print(fileDir)
    
    psyData = pd.read_csv(fileDir+[file for file in files_subject if file.endswith(r'.csv')][0])
    bool_trials = [isinstance(i, str) for i in list(psyData.written_scenario)] # select just the rows in the output file of the trials
    #select the variables needed form the csv file
    sel_vars = ['trial.thisN','participant','Condition','N_Scenario',
                'A1pos', 'A2pos','A3pos','B1pos', 'B2pos','B3pos','C1pos', 'C2pos','C3pos',
                'Imagesdecision_started',
                'decisiontimeA','decisiontimeB','decisiontimeC',
                'clicked_optionA','clicked_optionB','clicked_optionC',
                'slider_confidence_response',
                'slider_valueA_response','slider_valueB_response','slider_valueC_response',
                'slider_diffA_response','slider_diffB_response','slider_diffC_response']
    trials_all = psyData.loc[bool_trials, sel_vars].reset_index(drop=True)
    
    # Deselect training trials 
    trials = trials_all[2:].reset_index(drop=True) 
    
# Define new variables

    #Indicates which category is first choosen, second, third
    trials.loc[(trials['Condition']== 3),'decisiontimetotal']= trials[['decisiontimeA','decisiontimeB','decisiontimeC']].max(axis=1)
    trials.loc[(trials['Condition']== 3),'decisiontime1']= trials[['decisiontimeA','decisiontimeB','decisiontimeC']].min(axis=1) 
    trials.loc[(trials['Condition']== 3),'decisiontime2']= trials[['decisiontimeA','decisiontimeB','decisiontimeC']].apply(lambda x: x.nlargest(2).iloc[1], axis=1)

    # time frame where images are presented
    trials['Imagesdecision_start'] = trials['Imagesdecision_started'] 
    trials['Imagesdecision_stopped'] = trials['Imagesdecision_started'] +  trials['decisiontimetotal']
    #RT means the reaction time 
    #RT1 from start of simuli till the 1st decision
    trials['Imagesdecision_start1'] = trials['Imagesdecision_started'] 
    trials['Imagesdecision_stopped1'] = trials['Imagesdecision_started'] +  trials['decisiontime1']
    trials['RT1']=trials['Imagesdecision_stopped1']-trials['Imagesdecision_start1']
    
    #RT2 from first decision till the 2st decision
    trials['Imagesdecision_start2'] = trials['Imagesdecision_started'] +  trials['decisiontime1']
    trials['Imagesdecision_stopped2'] = trials['Imagesdecision_started'] +  trials['decisiontime2']
    trials['RT2']=trials['Imagesdecision_stopped2']-trials['Imagesdecision_start2']
    
    #RT3 from second decision till the 3st decision
    trials['Imagesdecision_start3'] = trials['Imagesdecision_started'] +  trials['decisiontime2']
    trials['Imagesdecision_stopped3'] = trials['Imagesdecision_started'] +  trials['decisiontimetotal']
    trials['RT3']=trials['Imagesdecision_stopped3']-trials['Imagesdecision_start3']
    # Change name of image location coordinates to Image locations according to our convention (image  at the top at 12 o'clock is number 3_1, 13 o'clock is number 3_2 and so on) the number 3 at the beginning indicates these are locations for 3x3 scenrios
    trials.replace({'[0.07, 0.393]': '3_1','[0.306, 0.255]': '3_2','[0.399, -0.003]': '3_3','[0.304, -0.26]': '3_4','[0.066, -0.395]': '3_5','[-0.203, -0.347]': '3_6','[-0.378, -0.136]': '3_7','[-0.377, 0.138]': '3_8','[-0.2, 0.346]': '3_9'}, inplace=True)
    
    
# Eyetracking data
    #get file
    etData = pd.DataFrame(np.array(h5py.File(fileDir+[file for file in files_subject if file.endswith(r'.hdf5')][0])['data_collection']['events']['eyetracker']['BinocularEyeSampleEvent']))
    
    #use function to detemrine exact moment where the eyetracking data is needed in each trial
    trials['et_decision_start'] = [closest(list(etData['time']),t) for t in trials['Imagesdecision_start']]
    trials['et_decision_end'] = [closest(list(etData['time']),t) for t in trials['Imagesdecision_stopped']]
   
    etData =  etData[['time','left_gaze_x', 'left_gaze_y','right_gaze_x', 'right_gaze_y']]
    
    #if only one eye take this value for both eyes since they fixate one point on the screen
    etData.loc[etData['left_gaze_x'].isna(),'left_gaze_x'] = etData['right_gaze_x']
    etData.loc[etData['left_gaze_y'].isna(),'left_gaze_y'] = etData['right_gaze_y']
    
    etData.loc[etData['right_gaze_x'].isna(),'right_gaze_x'] = etData['left_gaze_x']
    etData.loc[etData['right_gaze_y'].isna(),'right_gaze_y'] = etData['left_gaze_x']
     
    #average left and right eye
    etData['lr_x']= (etData ['left_gaze_x']+ etData ['right_gaze_x'])/2
    etData['lr_y']= (etData ['left_gaze_y']+ etData ['right_gaze_y'])/2
    
    #define regions of interest (1 = participant looked at that region at this time point)coordinates correspond to squares at the image locations
    etData.loc[etData['lr_x'].between(-0.11,0.25) & etData['lr_y'].between(0.21,0.573), '3_1'] = 1
    etData.loc[etData['lr_x'].between(0.12, 0.43) & etData['lr_y'].between(0.075,0.435), '3_2'] = 1
    etData.loc[etData['lr_x'].between(0.219,0.579) & etData['lr_y'].between(-0.183,0.177), '3_3'] = 1
    etData.loc[etData['lr_x'].between(0.124,0.484) & etData['lr_y'].between(-0.43,-0.08), '3_4'] = 1
    etData.loc[etData['lr_x'].between(-0.114,0.246) & etData['lr_y'].between(-0.575,-0.215), '3_5'] = 1
    etData.loc[etData['lr_x'].between(-0.383, -0.023) & etData['lr_y'].between(-0.527,-0.167), '3_6'] = 1
    etData.loc[etData['lr_x'].between(-0.558,-0.198) & etData['lr_y'].between(-0.316,0.044), '3_7'] = 1
    etData.loc[etData['lr_x'].between(-0.38, -0.02) & etData['lr_y'].between(-0.042,0.318), '3_8'] = 1
    etData.loc[etData['lr_x'].between(-0.38,-0.02) & etData['lr_y'].between(0.166,0.52), '3_9'] = 1   

    #define one column which gives you which region of interest is looked at = roi
    etData.loc[(etData['3_1']== 1),'roi'] = 1
    etData.loc[(etData['3_2']== 1),'roi'] = 2
    etData.loc[(etData['3_3']== 1),'roi'] = 3
    etData.loc[(etData['3_4']== 1),'roi'] = 4
    etData.loc[(etData['3_5']== 1),'roi'] = 5
    etData.loc[(etData['3_6']== 1),'roi'] = 6
    etData.loc[(etData['3_7']== 1),'roi'] = 7
    etData.loc[(etData['3_8']== 1),'roi'] = 8
    etData.loc[(etData['3_9']== 1),'roi'] = 9

    return(trials, etData)

def get_planning_data0(fileDir,subj):
    
    global trials
    global etData 
    files_subject = [_ for _ in os.listdir(fileDir) if (_.endswith(r".hdf5") or _.endswith(r".csv")) and "P"+str(subj) in _]
    print(fileDir)
    
    psyData = pd.read_csv(fileDir+[file for file in files_subject if file.endswith(r'.csv')][0])
    bool_trials = [isinstance(i, str) for i in list(psyData.written_scenario)] # select just the rows in the output file of the trials
    sel_vars = ['trial.thisN','participant','Condition','N_Scenario',
                'A1pos', 'A2pos','A3pos','B1pos', 'B2pos','B3pos','C1pos', 'C2pos','C3pos',
                'Imagesdecision_started',
                'decisiontimeA','decisiontimeB','decisiontimeC',
                'clicked_optionA','clicked_optionB','clicked_optionC',
                'slider_confidence_response',
                'slider_valueA_response','slider_valueB_response','slider_valueC_response',
                'slider_diffA_response','slider_diffB_response','slider_diffC_response']

    
    trials_all = psyData.loc[bool_trials, sel_vars].reset_index(drop=True)
    trials = trials_all[2:].reset_index(drop=True) #deselect training trials 
    
    # Define new variables  
    trials.loc[(trials['Condition']== 3),'decisiontimetotal']= trials[['decisiontimeA','decisiontimeB','decisiontimeC']].max(axis=1)
    trials.loc[(trials['Condition']== 3),'decisiontime1']= trials[['decisiontimeA','decisiontimeB','decisiontimeC']].min(axis=1) 
    trials.loc[(trials['Condition']== 3),'decisiontime2']= trials[['decisiontimeA','decisiontimeB','decisiontimeC']].apply(lambda x: x.nlargest(2).iloc[1], axis=1)

    # Define new time frame
    trials['Imagesdecision_start'] = trials['Imagesdecision_started'] 
    trials['Imagesdecision_stopped'] = trials['Imagesdecision_started'] +  trials['decisiontime1']


    #Change name of coordinates to Image locations 
    trials.replace({'[0.07, 0.393]': '3_1','[0.306, 0.255]': '3_2','[0.399, -0.003]': '3_3','[0.304, -0.26]': '3_4','[0.066, -0.395]': '3_5','[-0.203, -0.347]': '3_6','[-0.378, -0.136]': '3_7','[-0.377, 0.138]': '3_8','[-0.2, 0.346]': '3_9'}, inplace=True)
    
    
    # Eyetracking data
    etData = pd.DataFrame(np.array(h5py.File(fileDir+[file for file in files_subject if file.endswith(r'.hdf5')][0])['data_collection']['events']['eyetracker']['BinocularEyeSampleEvent']))
    
    trials['et_decision_start'] = [closest(list(etData['time']),t) for t in trials['Imagesdecision_start']]
    trials['et_decision_end'] = [closest(list(etData['time']),t) for t in trials['Imagesdecision_stopped']]
   
    etData =  etData[['time','left_gaze_x', 'left_gaze_y','right_gaze_x', 'right_gaze_y']]
    
    #if only one eye take this value for both eyes
    etData.loc[etData['left_gaze_x'].isna(),'left_gaze_x'] = etData['right_gaze_x']
    etData.loc[etData['left_gaze_y'].isna(),'left_gaze_y'] = etData['right_gaze_y']
    
    etData.loc[etData['right_gaze_x'].isna(),'right_gaze_x'] = etData['left_gaze_x']
    etData.loc[etData['right_gaze_y'].isna(),'right_gaze_y'] = etData['left_gaze_x']
   
    
    #average left and right eye
    etData['lr_x']= (etData ['left_gaze_x']+ etData ['right_gaze_x'])/2
    etData['lr_y']= (etData ['left_gaze_y']+ etData ['right_gaze_y'])/2
    
    #define regions of interest (1 = participant looked at that region at this time point)coordinates correspond to squares at the image locations
    etData.loc[etData['lr_x'].between(-0.11,0.25) & etData['lr_y'].between(0.21,0.573), '3_1'] = 1
    etData.loc[etData['lr_x'].between(0.12, 0.43) & etData['lr_y'].between(0.075,0.435), '3_2'] = 1
    etData.loc[etData['lr_x'].between(0.219,0.579) & etData['lr_y'].between(-0.183,0.177), '3_3'] = 1
    etData.loc[etData['lr_x'].between(0.124,0.484) & etData['lr_y'].between(-0.43,-0.08), '3_4'] = 1
    etData.loc[etData['lr_x'].between(-0.114,0.246) & etData['lr_y'].between(-0.575,-0.215), '3_5'] = 1
    etData.loc[etData['lr_x'].between(-0.383, -0.023) & etData['lr_y'].between(-0.527,-0.167), '3_6'] = 1
    etData.loc[etData['lr_x'].between(-0.558,-0.198) & etData['lr_y'].between(-0.316,0.044), '3_7'] = 1
    etData.loc[etData['lr_x'].between(-0.38, -0.02) & etData['lr_y'].between(-0.042,0.318), '3_8'] = 1
    etData.loc[etData['lr_x'].between(-0.38,-0.02) & etData['lr_y'].between(0.166,0.52), '3_9'] = 1   
#define one column which gives you which region of interest is looked at 
    etData.loc[(etData['3_1']== 1),'roi'] = 1
    etData.loc[(etData['3_2']== 1),'roi'] = 2
    etData.loc[(etData['3_3']== 1),'roi'] = 3
    etData.loc[(etData['3_4']== 1),'roi'] = 4
    etData.loc[(etData['3_5']== 1),'roi'] = 5
    etData.loc[(etData['3_6']== 1),'roi'] = 6
    etData.loc[(etData['3_7']== 1),'roi'] = 7
    etData.loc[(etData['3_8']== 1),'roi'] = 8
    etData.loc[(etData['3_9']== 1),'roi'] = 9

    return(trials, etData)

def get_planning_data1(fileDir,subj):

    global trials
    global etData 
    files_subject = [_ for _ in os.listdir(fileDir) if (_.endswith(r".hdf5") or _.endswith(r".csv")) and "P"+str(subj) in _]
    print(fileDir)
    
    psyData = pd.read_csv(fileDir+[file for file in files_subject if file.endswith(r'.csv')][0])
    bool_trials = [isinstance(i, str) for i in list(psyData.written_scenario)] # select just the rows in the output file of the trials
    sel_vars = ['trial.thisN','participant','Condition','N_Scenario',
                'A1pos', 'A2pos','A3pos','B1pos', 'B2pos','B3pos','C1pos', 'C2pos','C3pos',
                'Imagesdecision_started',
                'decisiontimeA','decisiontimeB','decisiontimeC',
                'clicked_optionA','clicked_optionB','clicked_optionC',
                'slider_confidence_response',
                'slider_valueA_response','slider_valueB_response','slider_valueC_response',
                'slider_diffA_response','slider_diffB_response','slider_diffC_response']

    
    trials_all = psyData.loc[bool_trials, sel_vars].reset_index(drop=True)
    trials = trials_all[2:].reset_index(drop=True) #deselect training trials 
    
    # Define new variables  
    trials.loc[(trials['Condition']== 3),'decisiontimetotal']= trials[['decisiontimeA','decisiontimeB','decisiontimeC']].max(axis=1)
    trials.loc[(trials['Condition']== 3),'decisiontime1']= trials[['decisiontimeA','decisiontimeB','decisiontimeC']].min(axis=1) 
    trials.loc[(trials['Condition']== 3),'decisiontime2']= trials[['decisiontimeA','decisiontimeB','decisiontimeC']].apply(lambda x: x.nlargest(2).iloc[1], axis=1)

    # Define new time frame
    trials['Imagesdecision_start'] = trials['Imagesdecision_started'] +  trials['decisiontime1']
    trials['Imagesdecision_stopped'] = trials['Imagesdecision_started'] +  trials['decisiontime2']


    #Change name of coordinates to Image locations 
    trials.replace({'[0.07, 0.393]': '3_1','[0.306, 0.255]': '3_2','[0.399, -0.003]': '3_3','[0.304, -0.26]': '3_4','[0.066, -0.395]': '3_5','[-0.203, -0.347]': '3_6','[-0.378, -0.136]': '3_7','[-0.377, 0.138]': '3_8','[-0.2, 0.346]': '3_9'}, inplace=True)
    
    
    # Eyetracking data
    etData = pd.DataFrame(np.array(h5py.File(fileDir+[file for file in files_subject if file.endswith(r'.hdf5')][0])['data_collection']['events']['eyetracker']['BinocularEyeSampleEvent']))
    
    trials['et_decision_start'] = [closest(list(etData['time']),t) for t in trials['Imagesdecision_start']]
    trials['et_decision_end'] = [closest(list(etData['time']),t) for t in trials['Imagesdecision_stopped']]
   
    etData =  etData[['time','left_gaze_x', 'left_gaze_y','right_gaze_x', 'right_gaze_y']]
    
    #if only one eye take this value for both eyes
    etData.loc[etData['left_gaze_x'].isna(),'left_gaze_x'] = etData['right_gaze_x']
    etData.loc[etData['left_gaze_y'].isna(),'left_gaze_y'] = etData['right_gaze_y']
    
    etData.loc[etData['right_gaze_x'].isna(),'right_gaze_x'] = etData['left_gaze_x']
    etData.loc[etData['right_gaze_y'].isna(),'right_gaze_y'] = etData['left_gaze_x']
   
    
    #average left and right eye
    etData['lr_x']= (etData ['left_gaze_x']+ etData ['right_gaze_x'])/2
    etData['lr_y']= (etData ['left_gaze_y']+ etData ['right_gaze_y'])/2
    
    #define regions of interest (1 = looked at region)

    etData.loc[etData['lr_x'].between(-0.01999999999999999,0.16) & etData['lr_y'].between(0.30300000000000005,0.4830), '3_1'] = 1
    etData.loc[etData['lr_x'].between(0.216, 0.396) & etData['lr_y'].between(0.165,0.345), '3_2'] = 1
    etData.loc[etData['lr_x'].between(0.30900000000000005, 0.489) & etData['lr_y'].between(-0.093,0.087), '3_3'] = 1
    etData.loc[etData['lr_x'].between(0.214, 0.394) & etData['lr_y'].between(-0.35,-0.17), '3_4'] = 1
    etData.loc[etData['lr_x'].between(-0.023999999999999994, 0.156) & etData['lr_y'].between(-0.485,-0.30500000000000005), '3_5'] = 1
    etData.loc[etData['lr_x'].between(-0.29300000000000004, -0.11300000000000002) & etData['lr_y'].between(-0.43699999999999994,-0.257), '3_6'] = 1
    etData.loc[etData['lr_x'].between(-0.46799999999999997,-0.28800000000000003) & etData['lr_y'].between(-0.226,-0.04600000000000001), '3_7'] = 1
    etData.loc[etData['lr_x'].between(-0.46699999999999997, -0.28700000000000003) & etData['lr_y'].between(0.048000000000000015,0.228), '3_8'] = 1
    etData.loc[etData['lr_x'].between(-0.29000000000000004,-0.11000000000000001) & etData['lr_y'].between(0.256,0.43599999999999994), '3_9'] = 1   

    #define one column which gives you which region of interest is looked at 
    etData.loc[(etData['3_1']== 1),'roi'] = 1
    etData.loc[(etData['3_2']== 1),'roi'] = 2
    etData.loc[(etData['3_3']== 1),'roi'] = 3
    etData.loc[(etData['3_4']== 1),'roi'] = 4
    etData.loc[(etData['3_5']== 1),'roi'] = 5
    etData.loc[(etData['3_6']== 1),'roi'] = 6
    etData.loc[(etData['3_7']== 1),'roi'] = 7
    etData.loc[(etData['3_8']== 1),'roi'] = 8
    etData.loc[(etData['3_9']== 1),'roi'] = 9

    return(trials, etData)

def get_planning_data2(fileDir,subj):

    global trials
    global etData 
    files_subject = [_ for _ in os.listdir(fileDir) if (_.endswith(r".hdf5") or _.endswith(r".csv")) and "P"+str(subj) in _]
    print(fileDir)
    
    psyData = pd.read_csv(fileDir+[file for file in files_subject if file.endswith(r'.csv')][0])
    bool_trials = [isinstance(i, str) for i in list(psyData.written_scenario)] # select just the rows in the output file of the trials
    sel_vars = ['trial.thisN','participant','Condition','N_Scenario',
                'A1pos', 'A2pos','A3pos','B1pos', 'B2pos','B3pos','C1pos', 'C2pos','C3pos',
                'Imagesdecision_started',
                'decisiontimeA','decisiontimeB','decisiontimeC',
                'clicked_optionA','clicked_optionB','clicked_optionC',
                'slider_confidence_response',
                'slider_valueA_response','slider_valueB_response','slider_valueC_response',
                'slider_diffA_response','slider_diffB_response','slider_diffC_response']

    
    trials_all = psyData.loc[bool_trials, sel_vars].reset_index(drop=True)
    trials = trials_all[2:].reset_index(drop=True) #deselect training trials 
    
    # Define new variables  
    trials.loc[(trials['Condition']== 3),'decisiontimetotal']= trials[['decisiontimeA','decisiontimeB','decisiontimeC']].max(axis=1)
    trials.loc[(trials['Condition']== 3),'decisiontime1']= trials[['decisiontimeA','decisiontimeB','decisiontimeC']].min(axis=1) 
    trials.loc[(trials['Condition']== 3),'decisiontime2']= trials[['decisiontimeA','decisiontimeB','decisiontimeC']].apply(lambda x: x.nlargest(2).iloc[1], axis=1)

    # Define new time frame
    trials['Imagesdecision_start'] = trials['Imagesdecision_started'] +  trials['decisiontime2']
    trials['Imagesdecision_stopped'] = trials['Imagesdecision_started'] +  trials['decisiontimetotal']


    #Change name of coordinates to Image locations 
    trials.replace({'[0.07, 0.393]': '3_1','[0.306, 0.255]': '3_2','[0.399, -0.003]': '3_3','[0.304, -0.26]': '3_4','[0.066, -0.395]': '3_5','[-0.203, -0.347]': '3_6','[-0.378, -0.136]': '3_7','[-0.377, 0.138]': '3_8','[-0.2, 0.346]': '3_9'}, inplace=True)
    
    
    # Eyetracking data
    etData = pd.DataFrame(np.array(h5py.File(fileDir+[file for file in files_subject if file.endswith(r'.hdf5')][0])['data_collection']['events']['eyetracker']['BinocularEyeSampleEvent']))
    
    trials['et_decision_start'] = [closest(list(etData['time']),t) for t in trials['Imagesdecision_start']]
    trials['et_decision_end'] = [closest(list(etData['time']),t) for t in trials['Imagesdecision_stopped']]
   
    etData =  etData[['time','left_gaze_x', 'left_gaze_y','right_gaze_x', 'right_gaze_y']]
    
    #if only one eye take this value for both eyes
    etData.loc[etData['left_gaze_x'].isna(),'left_gaze_x'] = etData['right_gaze_x']
    etData.loc[etData['left_gaze_y'].isna(),'left_gaze_y'] = etData['right_gaze_y']
    
    etData.loc[etData['right_gaze_x'].isna(),'right_gaze_x'] = etData['left_gaze_x']
    etData.loc[etData['right_gaze_y'].isna(),'right_gaze_y'] = etData['left_gaze_x']
   
    
    #average left and right eye
    etData['lr_x']= (etData ['left_gaze_x']+ etData ['right_gaze_x'])/2
    etData['lr_y']= (etData ['left_gaze_y']+ etData ['right_gaze_y'])/2
    
    #define regions of interest (1 = looked at region)
    #define regions of interest (1 = participant looked at that region at this time point)coordinates correspond to squares at the image locations
    etData.loc[etData['lr_x'].between(-0.11,0.25) & etData['lr_y'].between(0.21,0.573), '3_1'] = 1
    etData.loc[etData['lr_x'].between(0.12, 0.43) & etData['lr_y'].between(0.075,0.435), '3_2'] = 1
    etData.loc[etData['lr_x'].between(0.219,0.579) & etData['lr_y'].between(-0.183,0.177), '3_3'] = 1
    etData.loc[etData['lr_x'].between(0.124,0.484) & etData['lr_y'].between(-0.43,-0.08), '3_4'] = 1
    etData.loc[etData['lr_x'].between(-0.114,0.246) & etData['lr_y'].between(-0.575,-0.215), '3_5'] = 1
    etData.loc[etData['lr_x'].between(-0.383, -0.023) & etData['lr_y'].between(-0.527,-0.167), '3_6'] = 1
    etData.loc[etData['lr_x'].between(-0.558,-0.198) & etData['lr_y'].between(-0.316,0.044), '3_7'] = 1
    etData.loc[etData['lr_x'].between(-0.38, -0.02) & etData['lr_y'].between(-0.042,0.318), '3_8'] = 1
    etData.loc[etData['lr_x'].between(-0.38,-0.02) & etData['lr_y'].between(0.166,0.52), '3_9'] = 1   

   #define one column which gives you which region of interest is looked at 
    etData.loc[(etData['3_1']== 1),'roi'] = 1
    etData.loc[(etData['3_2']== 1),'roi'] = 2
    etData.loc[(etData['3_3']== 1),'roi'] = 3
    etData.loc[(etData['3_4']== 1),'roi'] = 4
    etData.loc[(etData['3_5']== 1),'roi'] = 5
    etData.loc[(etData['3_6']== 1),'roi'] = 6
    etData.loc[(etData['3_7']== 1),'roi'] = 7
    etData.loc[(etData['3_8']== 1),'roi'] = 8
    etData.loc[(etData['3_9']== 1),'roi'] = 9

    return(trials, etData)

def select_eyedata(trial,index,etData):
    global invalid_datapoints
    
    #select eyedata for the trial
    gazedata = etData.loc[(etData['time']>=trial['et_decision_start']) & (etData['time']<=trial['et_decision_end']), ['time','left_gaze_x', 'left_gaze_y','right_gaze_x', 'right_gaze_y','lr_x','lr_y','3_1','3_2','3_3','3_4','3_5','3_6','3_7','3_8','3_9','roi']]
   
    # remove rows with no data
    gazedata[([isnan(i) for i in list(gazedata.left_gaze_x)]) and ([isnan(i) for i in list(gazedata.right_gaze_x)])] = float('nan')
    
    #if true sampling point is invalid
    gazedata['invalid']= gazedata['lr_x'].isna() | gazedata['lr_y'].isna()

    #within trial % of invalid datapoints
    gazedata['invalid'] = etData['lr_x'].isna() & etData['lr_y'].isna() # invalid data point = True
    invalid_datapoints = gazedata.invalid.value_counts()
    invalid_datapoints = gazedata.invalid.value_counts(True)
    invalid_datapoints = invalid_datapoints.tolist()
    return(gazedata)  


def prepare_csv(): #function transforms eyedata to csv information
    global trials
    
#set up empty lists which will be transformed into pandas df later
    list3_1=[]
    list3_2=[]
    list3_3=[]
    list3_4=[]
    list3_5=[]
    list3_6=[]
    list3_7=[]
    list3_8=[]
    list3_9=[]
    invalid_trials =[]

    #loop for each trial seperate eye data
    for index, trial in trials.iterrows():
        # select eyetracking data for the trial:      
        gazedata = select_eyedata(trial,index,etData)
        invalid_trials.append(invalid_datapoints[::2])
        
        #calculate eye data point sum for each region of interest and trial
        list3_1.append(gazedata['3_1'].sum())
        list3_2.append(gazedata['3_2'].sum())
        list3_3.append(gazedata['3_3'].sum())
        list3_4.append(gazedata['3_4'].sum())
        list3_5.append(gazedata['3_5'].sum())
        list3_6.append(gazedata['3_6'].sum())
        list3_7.append(gazedata['3_7'].sum())
        list3_8.append(gazedata['3_8'].sum())
        list3_9.append(gazedata['3_9'].sum())
    
    #get valid datapoint fraction for each trial
    flat_invalid_trials = [item for sublist in invalid_trials for item in sublist]
    trials['valid_datapoints'] = flat_invalid_trials  
    trials['valid_datapoints'] =(trials['valid_datapoints'])*100
    

    # add number of region of interest gaze points (n) to csv per trial
    trials['n_3_1']= list3_1
    trials['n_3_2']= list3_2
    trials['n_3_3']= list3_3
    trials['n_3_4']= list3_4
    trials['n_3_5']= list3_5
    trials['n_3_6']= list3_6
    trials['n_3_7']= list3_7
    trials['n_3_8']= list3_8
    trials['n_3_9']= list3_9
    
    #So far we have the looking time at locations now we transform it to looking time at images: Where was image A1 presented? Take this location and use its looking time for A1
    start = [3]
    end_3 = [1,2,3,4,5,6,7,8,9]
    
    for start_n in start:
         if start_n == 3:
             end = end_3
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['A1pos']==str(start_n)+'_'+str(end_n),['n_A1']] = trials.loc[trials['A1pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['A2pos']==str(start_n)+'_'+str(end_n),['n_A2']] = trials.loc[trials['A2pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['A3pos']==str(start_n)+'_'+str(end_n),['n_A3']] = trials.loc[trials['A3pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['B1pos']==str(start_n)+'_'+str(end_n),['n_B1']] = trials.loc[trials['B1pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values           
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['B2pos']==str(start_n)+'_'+str(end_n),['n_B2']] = trials.loc[trials['B2pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['B3pos']==str(start_n)+'_'+str(end_n),['n_B3']] = trials.loc[trials['B3pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['C1pos']==str(start_n)+'_'+str(end_n),['n_C1']] = trials.loc[trials['C1pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['C2pos']==str(start_n)+'_'+str(end_n),['n_C2']] = trials.loc[trials['C2pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values           
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['C3pos']==str(start_n)+'_'+str(end_n),['n_C3']] = trials.loc[trials['C3pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values               
                    
    # change data points to time  (120 is the sampling rate per second)
    trials['t_A1'] = trials['n_A1']/120
    trials['t_A2'] = trials['n_A2']/120
    trials['t_A3'] = trials['n_A3']/120
    trials['t_B1'] = trials['n_B1']/120
    trials['t_B2'] = trials['n_B2']/120
    trials['t_B3'] = trials['n_B3']/120
    trials['t_C1'] = trials['n_C1']/120
    trials['t_C2'] = trials['n_C2']/120
    trials['t_C3'] = trials['n_C3']/120       

    # Add looking time at images to obtain looking time at each category
    trials.loc[(trials['Condition']== 3),'t_categoryA'] = trials['t_A1']+ trials['t_A2']+ trials['t_A3']
    trials.loc[(trials['Condition']== 3),'t_categoryB'] = trials['t_B1']+ trials['t_B2']+ trials['t_B3']
    trials.loc[(trials['Condition']== 3),'t_categoryC'] = trials['t_C1']+ trials['t_C2']+ trials['t_C3']

   # get fraction of looking time per category
    trials.loc[(trials['Condition']== 3),'t_cat_total'] = trials['t_categoryA']+ trials['t_categoryB']+ trials['t_categoryC']
   
    trials['f_categoryA']= trials['t_categoryA']/trials['t_cat_total']
    trials['f_categoryB']= trials['t_categoryB']/trials['t_cat_total']
    trials['f_categoryC']= trials['t_categoryC']/trials['t_cat_total']
    
    #change slider values to standart ones by substracting the mean dificulty or importance of the trial
    trials.loc[(trials['Condition']== 3),'imp_mean_cat'] = trials[['slider_valueA_response', 'slider_valueB_response', 'slider_valueC_response']].mean(axis=1, skipna=True)
    trials['slider_valueA_response']= trials['slider_valueA_response']
    trials['slider_valueB_response']= trials['slider_valueB_response']
    trials['slider_valueC_response']= trials['slider_valueC_response']
    
    trials.loc[(trials['Condition']== 3),'diff_mean_cat'] = trials[['slider_diffA_response', 'slider_diffB_response', 'slider_diffC_response']].mean(axis=1, skipna=True)
    trials['slider_diffA_response']= trials['slider_diffA_response']
    trials['slider_diffB_response']= trials['slider_diffB_response']
    trials['slider_diffC_response']= trials['slider_diffC_response']
    

# Gaze switching 

# Create lists of positions which belong to the same category
    C1 = [1,4,7]
    C2 = [2,5,8]
    C3 = [3,6,9]
    
    #empty lists for number of switches to append to trials df
    B=[] #B = Between  category switch
    WC1=[] #WC1 within category 1 switch
    WC2=[] #WC1 within category 2 switch
    WC3=[] #WC1 within category 3 switch
    
   
    global gazedata_alltrials
    gazedata_alltrials =[]
    
    #loop for each trial seperate eye data
    for index, trial in trials.iterrows():
        # select eyetracking data for the trial and store it in list gazedata of all trials       
        gazedata = select_eyedata(trial,index,etData)
        gazedata_alltrials.append(gazedata)
        
        #Create a seperate dataframe from the column roi (which region of interest is looked at)
        roi = pd.DataFrame(gazedata,columns=['roi']).reset_index(drop=True)
        roi = roi.dropna().reset_index(drop=True)
 
        # Go through roi and check when the region of interest looked at changes if it is a between or within switch
        prev_roi = roi.shift()
        roi['Change'] = np.select([(roi == prev_roi) | prev_roi.isnull(),
        roi.isin(C1) & prev_roi.isin(C1),
        roi.isin(C2) & prev_roi.isin(C2),
        roi.isin(C3) & prev_roi.isin(C3),
        roi != prev_roi],[ "nan", "C1", "C2", "C3", "B",],)
    
        #new dataframe of both
        switch = pd.DataFrame(columns=['subject','Change'])
        switch['Change'] =  roi['Change']
        switch = switch[switch['Change'].str.contains("nan")==False]
          
        B.append((switch['Change'] == 'B').sum()) #append for each trial how many switches were performed between categoires
        WC1.append((switch['Change'] == 'C1').sum()) #append for each trial how many switches were performed within  categoires
        WC2.append((switch['Change'] == 'C2').sum()) 
        WC3.append((switch['Change'] == 'C3').sum()) 
        
        #add the obtained results to our trials dataframe
    trials['WC1']= WC1
    trials['WC2']= WC2
    trials['WC3']= WC3
    trials['B']= B
    
    #transform values to fractions 
    trials['W'] = trials['WC1']+trials['WC2']+trials['WC3']
    trials['W_B_total'] =  trials['W'] +  trials['B']
    trials['W'] = trials['W']/trials['W_B_total']
    trials['B'] = trials['B']/trials['W_B_total']
    
    trials['WC1']=trials['WC1']/trials['W']
    trials['WC2']=trials['WC2']/trials['W']
    trials['WC3']=trials['WC3']/trials['W']
    

#SLIDER RATINGS AND DECISON ORDER
   
    #get name of 1,2,3 category based on decision-time for each condition
    trials.loc[(trials['Condition']== 3),'decision1_category'] = trials[['decisiontimeA','decisiontimeB','decisiontimeC']].idxmin(axis = 1)
    trials.loc[(trials['Condition']== 3),'decision2_category'] = trials[['decisiontimeA','decisiontimeB','decisiontimeC']].T.apply(lambda x: x.nlargest(2).idxmin())
    trials.loc[(trials['Condition']== 3),'decision3_category'] = trials[['decisiontimeA','decisiontimeB','decisiontimeC']].idxmax(axis = 1)
      
    trials.loc[trials['decision1_category']=='decisiontimeA','decision1_category'] = 'A'
    trials.loc[trials['decision1_category']=='decisiontimeB','decision1_category'] = 'B'
    trials.loc[trials['decision1_category']=='decisiontimeC','decision1_category'] = 'C'
    
    trials.loc[trials['decision2_category']=='decisiontimeA','decision2_category'] = 'A'
    trials.loc[trials['decision2_category']=='decisiontimeB','decision2_category'] = 'B'
    trials.loc[trials['decision2_category']=='decisiontimeC','decision2_category'] = 'C'
    
    trials.loc[trials['decision3_category']=='decisiontimeA','decision3_category'] = 'A'
    trials.loc[trials['decision3_category']=='decisiontimeB','decision3_category'] = 'B'
    trials.loc[trials['decision3_category']=='decisiontimeC','decision3_category'] = 'C'
    
    #get importance value for 1,2,3 choosen category
    trials.loc[trials['decision1_category']=='A','category1_imp'] = trials['slider_valueA_response']
    trials.loc[trials['decision1_category']=='B','category1_imp'] = trials['slider_valueB_response']
    trials.loc[trials['decision1_category']=='C','category1_imp'] = trials['slider_valueC_response']
    
    trials.loc[trials['decision2_category']=='A','category2_imp'] = trials['slider_valueA_response']
    trials.loc[trials['decision2_category']=='B','category2_imp'] = trials['slider_valueB_response']
    trials.loc[trials['decision2_category']=='C','category2_imp'] = trials['slider_valueC_response']
    
    trials.loc[trials['decision3_category']=='A','category3_imp'] = trials['slider_valueA_response']
    trials.loc[trials['decision3_category']=='B','category3_imp'] = trials['slider_valueB_response']
    trials.loc[trials['decision3_category']=='C','category3_imp'] = trials['slider_valueC_response']
    
    # #get difficulty value for 1,2,3 choosen category
    trials.loc[trials['decision1_category']=='A','category1_diff'] = trials['slider_diffA_response']
    trials.loc[trials['decision1_category']=='B','category1_diff'] = trials['slider_diffB_response']   
    trials.loc[trials['decision1_category']=='C','category1_diff'] = trials['slider_diffC_response'] 
    
    trials.loc[trials['decision2_category']=='A','category2_diff'] = trials['slider_diffA_response']
    trials.loc[trials['decision2_category']=='B','category2_diff'] = trials['slider_diffB_response']   
    trials.loc[trials['decision2_category']=='C','category2_diff'] = trials['slider_diffC_response']   
    
    trials.loc[trials['decision3_category']=='A','category3_diff'] = trials['slider_diffA_response']
    trials.loc[trials['decision3_category']=='B','category3_diff'] = trials['slider_diffB_response']   
    trials.loc[trials['decision3_category']=='C','category3_diff'] = trials['slider_diffC_response']   

    #drop columns which are not needed anymore
    trials = trials.drop(labels = ['t_categoryA','t_categoryB','t_categoryC','n_A1','n_A2','n_A3','n_B1','n_B2','n_B3','n_C1','n_C2','n_C3','diff_mean_cat','imp_mean_cat','slider_valueA_response','slider_valueB_response','slider_valueC_response','slider_diffA_response','slider_diffB_response','slider_diffC_response'], axis = 1).reset_index(drop=True)  
    
    #get looking time for 1,2,3 choosen category
    trials.loc[trials['decision1_category']=='A','f_category1'] = trials['f_categoryA']
    trials.loc[trials['decision1_category']=='B','f_category1'] = trials['f_categoryB']   
    trials.loc[trials['decision1_category']=='C','f_category1'] = trials['f_categoryC'] 
    
    trials.loc[trials['decision2_category']=='A','f_category2'] = trials['f_categoryA']
    trials.loc[trials['decision2_category']=='B','f_category2'] = trials['f_categoryB']   
    trials.loc[trials['decision2_category']=='C','f_category2'] = trials['f_categoryC'] 
    
    trials.loc[trials['decision3_category']=='A','f_category3'] = trials['f_categoryA']
    trials.loc[trials['decision3_category']=='B','f_category3'] = trials['f_categoryB']   
    trials.loc[trials['decision3_category']=='C','f_category3'] = trials['f_categoryC'] 
    
#TRANSLATIONFOR GAZE DATA PLOTS 
    # We use the new dataframe V1 for our calculations    
    V1 = trials[['decision1_category','decision2_category','decision3_category','clicked_optionA','clicked_optionB','clicked_optionC','A1pos','A2pos','A3pos','B1pos','B2pos','B3pos','C1pos','C2pos','C3pos']] 
        
# For our gaze plots we order our options based on this convention: 
    #1_1 = Fist selected category and selected option
    #1_2 = Fist selected category and not selected option
    #1_3 = Fist selected category and not selected option
    #2_1 = Second selected category and selected option....
    
    
    # first we optain the seleted option from each category
    V1.loc[trials['decision1_category']=='A','decision1_1category'] = V1['clicked_optionA']
    V1.loc[trials['decision1_category']=='B','decision1_1category'] = V1['clicked_optionB']
    V1.loc[trials['decision1_category']=='C','decision1_1category'] = V1['clicked_optionC']
    
    V1.loc[trials['decision2_category']=='A','decision2_1category'] = V1['clicked_optionA']
    V1.loc[trials['decision2_category']=='B','decision2_1category'] = V1['clicked_optionB']
    V1.loc[trials['decision2_category']=='C','decision2_1category'] = V1['clicked_optionC']
    
    V1.loc[trials['decision3_category']=='A','decision3_1category'] = V1['clicked_optionA']
    V1.loc[trials['decision3_category']=='B','decision3_1category'] = V1['clicked_optionB']
    V1.loc[trials['decision3_category']=='C','decision3_1category'] = V1['clicked_optionC']
    
    V1 = V1.drop(labels = ['clicked_optionA','clicked_optionB','clicked_optionC','decision1_category','decision2_category','decision3_category'], axis = 1).reset_index(drop=True)
   
    # We drop the 3_ since all trials are 3x3 in this experiment 
    V1.A1pos = V1.A1pos.replace({'3_':''}, regex=True)
    V1.A2pos = V1.A2pos.replace({'3_':''}, regex=True)
    V1.A3pos = V1.A3pos.replace({'3_':''}, regex=True)
    V1.B1pos = V1.B1pos.replace({'3_':''}, regex=True)
    V1.B2pos = V1.B2pos.replace({'3_':''}, regex=True)
    V1.B3pos = V1.B3pos.replace({'3_':''}, regex=True)
    V1.C1pos = V1.C1pos.replace({'3_':''}, regex=True)
    V1.C2pos = V1.C2pos.replace({'3_':''}, regex=True)
    V1.C3pos = V1.C3pos.replace({'3_':''}, regex=True)
    
    cols = ['decision1_1category', 'decision2_1category', 'decision3_1category']
    V1[cols] = V1[cols].applymap(lambda lst: eval(lst)[0])
    
    #Based on the selected option we know which options are not selected and can assign them to our naming convention for the gaze plots
    V1.loc[V1['decision1_1category']=='A1','decision1_2category'] = 'A2'
    V1.loc[V1['decision1_1category']=='A1','decision1_3category'] = 'A3'
    
    V1.loc[V1['decision1_1category']=='A2','decision1_2category'] = 'A1'
    V1.loc[V1['decision1_1category']=='A2','decision1_3category'] = 'A3'
    
    V1.loc[V1['decision1_1category']=='A3','decision1_2category'] = 'A1'
    V1.loc[V1['decision1_1category']=='A3','decision1_3category'] = 'A2'
    
    
    V1.loc[V1['decision1_1category']=='B1','decision1_2category'] = 'B2'
    V1.loc[V1['decision1_1category']=='B1','decision1_3category'] = 'B3'
    
    V1.loc[V1['decision1_1category']=='B2','decision1_2category'] = 'B1'
    V1.loc[V1['decision1_1category']=='B2','decision1_3category'] = 'B3'
    
    V1.loc[V1['decision1_1category']=='B3','decision1_2category'] = 'B1'
    V1.loc[V1['decision1_1category']=='B3','decision1_3category'] = 'B2'
    
    
    V1.loc[V1['decision1_1category']=='C1','decision1_2category'] = 'C2'
    V1.loc[V1['decision1_1category']=='C1','decision1_3category'] = 'C3'
    
    V1.loc[V1['decision1_1category']=='C2','decision1_2category'] = 'C1'
    V1.loc[V1['decision1_1category']=='C2','decision1_3category'] = 'C3'
    
    V1.loc[V1['decision1_1category']=='C3','decision1_2category'] = 'C1'
    V1.loc[V1['decision1_1category']=='C3','decision1_3category'] = 'C2'
    
    
    
    V1.loc[V1['decision2_1category']=='A1','decision2_2category'] = 'A2'
    V1.loc[V1['decision2_1category']=='A1','decision2_3category'] = 'A3'
    
    V1.loc[V1['decision2_1category']=='A2','decision2_2category'] = 'A1'
    V1.loc[V1['decision2_1category']=='A2','decision2_3category'] = 'A3'
    
    V1.loc[V1['decision2_1category']=='A3','decision2_2category'] = 'A1'
    V1.loc[V1['decision2_1category']=='A3','decision2_3category'] = 'A2'
    
    
    V1.loc[V1['decision2_1category']=='B1','decision2_2category'] = 'B2'
    V1.loc[V1['decision2_1category']=='B1','decision2_3category'] = 'B3'
    
    V1.loc[V1['decision2_1category']=='B2','decision2_2category'] = 'B1'
    V1.loc[V1['decision2_1category']=='B2','decision2_3category'] = 'B3'
    
    V1.loc[V1['decision2_1category']=='B3','decision2_2category'] = 'B1'
    V1.loc[V1['decision2_1category']=='B3','decision2_3category'] = 'B2'
    
    
    V1.loc[V1['decision2_1category']=='C1','decision2_2category'] = 'C2'
    V1.loc[V1['decision2_1category']=='C1','decision2_3category'] = 'C3'
    
    V1.loc[V1['decision2_1category']=='C2','decision2_2category'] = 'C1'
    V1.loc[V1['decision2_1category']=='C2','decision2_3category'] = 'C3'
    
    V1.loc[V1['decision2_1category']=='C3','decision2_2category'] = 'C1'
    V1.loc[V1['decision2_1category']=='C3','decision2_3category'] = 'C2'



    V1.loc[V1['decision3_1category']=='A1','decision3_2category'] = 'A2'
    V1.loc[V1['decision3_1category']=='A1','decision3_3category'] = 'A3'
    
    V1.loc[V1['decision3_1category']=='A2','decision3_2category'] = 'A1'
    V1.loc[V1['decision3_1category']=='A2','decision3_3category'] = 'A3'
    
    V1.loc[V1['decision3_1category']=='A3','decision3_2category'] = 'A1'
    V1.loc[V1['decision3_1category']=='A3','decision3_3category'] = 'A2'

    
    V1.loc[V1['decision3_1category']=='B1','decision3_2category'] = 'B2'
    V1.loc[V1['decision3_1category']=='B1','decision3_3category'] = 'B3'
    
    V1.loc[V1['decision3_1category']=='B2','decision3_2category'] = 'B1'
    V1.loc[V1['decision3_1category']=='B2','decision3_3category'] = 'B3'
    
    V1.loc[V1['decision3_1category']=='B3','decision3_2category'] = 'B1'
    V1.loc[V1['decision3_1category']=='B3','decision3_3category'] = 'B2'
    
    
    V1.loc[V1['decision3_1category']=='C1','decision3_2category'] = 'C2'
    V1.loc[V1['decision3_1category']=='C1','decision3_3category'] = 'C3'
    
    V1.loc[V1['decision3_1category']=='C2','decision3_2category'] = 'C1'
    V1.loc[V1['decision3_1category']=='C2','decision3_3category'] = 'C3'
    
    V1.loc[V1['decision3_1category']=='C3','decision3_2category'] = 'C1'
    V1.loc[V1['decision3_1category']=='C3','decision3_3category'] = 'C2'
    
    x = ['decision1_1category','decision1_2category','decision1_3category','decision2_1category','decision2_2category','decision2_3category','decision3_1category','decision3_2category','decision3_3category']
    
    # now we know that for examle image A1 is the first selected option. But what position (region of interest) is it? here we add the position
    for i in x:
        V1.loc[V1[i]=='A1',i] = V1['A1pos']
        V1.loc[V1[i]=='A2',i] = V1['A2pos']
        V1.loc[V1[i]=='A3',i] = V1['A3pos']
        
        V1.loc[V1[i]=='B1',i] = V1['B1pos']
        V1.loc[V1[i]=='B2',i] = V1['B2pos']
        V1.loc[V1[i]=='B3',i] = V1['B3pos']
        
        V1.loc[V1[i]=='C1',i] = V1['C1pos']
        V1.loc[V1[i]=='C2',i] = V1['C2pos']
        V1.loc[V1[i]=='C3',i] = V1['C3pos']
  
    V1 = V1.drop(labels = ['A1pos','A2pos','A3pos','B1pos','B2pos','B3pos','C1pos','C2pos','C3pos'], axis = 1).reset_index(drop=True)
    #try this for ploting this gaze data
    
    trials['decision1_1category'] = V1['decision1_1category']
    trials['decision1_2category'] = V1['decision1_2category']
    trials['decision1_3category'] = V1['decision1_3category']
    
    trials['decision2_1category'] = V1['decision2_1category']
    trials['decision2_2category'] = V1['decision2_2category']
    trials['decision2_3category'] = V1['decision2_3category']
    
    trials['decision3_1category'] = V1['decision3_1category']
    trials['decision3_2category'] = V1['decision3_2category']
    trials['decision3_3category'] = V1['decision3_3category']
    
    #Add mean values across all trials of the participant to the results dataframe
    results.loc[results['participant']==trials['participant'].mean(),'validgaze_percent'] = trials['valid_datapoints'].mean()
    
    results.loc[results['participant']==trials['participant'].mean(),'W'] = trials['W'].mean()
    results.loc[results['participant']==trials['participant'].mean(),'B'] = trials['B'].mean()
    
    results.loc[results['participant']==trials['participant'].mean(),'f_category1'] = trials['f_category1'].mean()
    results.loc[results['participant']==trials['participant'].mean(),'f_category2'] = trials['f_category2'].mean()
    results.loc[results['participant']==trials['participant'].mean(),'f_category3'] = trials['f_category3'].mean()

    
    results.loc[results['participant']==trials['participant'].mean(),'decisiontime1'] = trials['decisiontime1'].mean()
    results.loc[results['participant']==trials['participant'].mean(),'decisiontime2'] = trials['decisiontime2'].mean()
    results.loc[results['participant']==trials['participant'].mean(),'decisiontimetotal'] = trials['decisiontimetotal'].mean()
    
    results.loc[results['participant']==trials['participant'].mean(),'RT1'] = trials['RT1'].mean()
    results.loc[results['participant']==trials['participant'].mean(),'RT2'] = trials['RT2'].mean()
    results.loc[results['participant']==trials['participant'].mean(),'RT3'] = trials['RT3'].mean()
    
    
    
    
    results.loc[results['participant']==trials['participant'].mean(),'category1_imp'] = trials['category1_imp'].mean()
    results.loc[results['participant']==trials['participant'].mean(),'category2_imp'] = trials['category2_imp'].mean()
    results.loc[results['participant']==trials['participant'].mean(),'category3_imp'] = trials['category3_imp'].mean()
    
    results.loc[results['participant']==trials['participant'].mean(),'category1_diff'] = trials['category1_diff'].mean()
    results.loc[results['participant']==trials['participant'].mean(),'category2_diff'] = trials['category2_diff'].mean()
    results.loc[results['participant']==trials['participant'].mean(),'category3_diff'] = trials['category3_diff'].mean()
    
    results.loc[results['participant']==trials['participant'].mean(),'WC1'] = trials['WC1'].mean()
    results.loc[results['participant']==trials['participant'].mean(),'WC2'] = trials['WC2'].mean()
    results.loc[results['participant']==trials['participant'].mean(),'WC3'] = trials['WC3'].mean()

#same function as above but for a specific time frame
def prepare_csv_0():
    global trials
    
#LOOKING TIME TO CSV
    list3_1=[]
    list3_2=[]
    list3_3=[]
    list3_4=[]
    list3_5=[]
    list3_6=[]
    list3_7=[]
    list3_8=[]
    list3_9=[]
    invalid_trials =[]

    #loop for each trial seperate eye data
    for index, trial in trials.iterrows():
        # select eyetracking data for the trial:      
        gazedata = select_eyedata(trial,index,etData)
        invalid_trials.append(invalid_datapoints[::2])
        
        #calculate eye data point sum for each region of interest and trial
        list3_1.append(gazedata['3_1'].sum())
        list3_2.append(gazedata['3_2'].sum())
        list3_3.append(gazedata['3_3'].sum())
        list3_4.append(gazedata['3_4'].sum())
        list3_5.append(gazedata['3_5'].sum())
        list3_6.append(gazedata['3_6'].sum())
        list3_7.append(gazedata['3_7'].sum())
        list3_8.append(gazedata['3_8'].sum())
        list3_9.append(gazedata['3_9'].sum())
    
    #get valid datapoint proportion for each trial
    flat_invalid_trials = [item for sublist in invalid_trials for item in sublist]
    trials['valid_datapoints'] = flat_invalid_trials  
    trials['valid_datapoints'] =(trials['valid_datapoints'])*100
    #trials = trials.drop(trials[trials['valid_trial']<0.25].index).reset_index(drop=True)
    

    # add number of region of interest gaze points to csv per trial
    trials['n_3_1']= list3_1
    trials['n_3_2']= list3_2
    trials['n_3_3']= list3_3
    trials['n_3_4']= list3_4
    trials['n_3_5']= list3_5
    trials['n_3_6']= list3_6
    trials['n_3_7']= list3_7
    trials['n_3_8']= list3_8
    trials['n_3_9']= list3_9
    
    # use location eye data to realte it to image looking time 
    start = [3]
    end_3 = [1,2,3,4,5,6,7,8,9]

    
    for start_n in start:
         if start_n == 3:
             end = end_3
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['A1pos']==str(start_n)+'_'+str(end_n),['n_A1']] = trials.loc[trials['A1pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['A2pos']==str(start_n)+'_'+str(end_n),['n_A2']] = trials.loc[trials['A2pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['A3pos']==str(start_n)+'_'+str(end_n),['n_A3']] = trials.loc[trials['A3pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['B1pos']==str(start_n)+'_'+str(end_n),['n_B1']] = trials.loc[trials['B1pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values           
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['B2pos']==str(start_n)+'_'+str(end_n),['n_B2']] = trials.loc[trials['B2pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['B3pos']==str(start_n)+'_'+str(end_n),['n_B3']] = trials.loc[trials['B3pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['C1pos']==str(start_n)+'_'+str(end_n),['n_C1']] = trials.loc[trials['C1pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['C2pos']==str(start_n)+'_'+str(end_n),['n_C2']] = trials.loc[trials['C2pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values           
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['C3pos']==str(start_n)+'_'+str(end_n),['n_C3']] = trials.loc[trials['C3pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values               
                    
    # change data points to time 
    trials['t_A1'] = trials['n_A1']/120
    trials['t_A2'] = trials['n_A2']/120
    trials['t_A3'] = trials['n_A3']/120
    trials['t_B1'] = trials['n_B1']/120
    trials['t_B2'] = trials['n_B2']/120
    trials['t_B3'] = trials['n_B3']/120
    trials['t_C1'] = trials['n_C1']/120
    trials['t_C2'] = trials['n_C2']/120
    trials['t_C3'] = trials['n_C3']/120       

    trials.loc[(trials['Condition']== 3),'t_categoryA'] = trials['t_A1']+ trials['t_A2']+ trials['t_A3']
    trials.loc[(trials['Condition']== 3),'t_categoryB'] = trials['t_B1']+ trials['t_B2']+ trials['t_B3']
    trials.loc[(trials['Condition']== 3),'t_categoryC'] = trials['t_C1']+ trials['t_C2']+ trials['t_C3']

   #get fraction of lookin time per category
    trials.loc[(trials['Condition']== 3),'t_cat_total'] = trials['t_categoryA']+ trials['t_categoryB']+ trials['t_categoryC']
   
    trials['f_categoryA']= trials['t_categoryA']/trials['t_cat_total']
    trials['f_categoryB']= trials['t_categoryB']/trials['t_cat_total']
    trials['f_categoryC']= trials['t_categoryC']/trials['t_cat_total']
    
    #change slider values to standart ones 
    trials.loc[(trials['Condition']== 3),'imp_mean_cat'] = trials[['slider_valueA_response', 'slider_valueB_response', 'slider_valueC_response']].mean(axis=1, skipna=True)
    trials['slider_valueA_response']= trials['slider_valueA_response']
    trials['slider_valueB_response']= trials['slider_valueB_response']
    trials['slider_valueC_response']= trials['slider_valueC_response']
    
    trials.loc[(trials['Condition']== 3),'diff_mean_cat'] = trials[['slider_diffA_response', 'slider_diffB_response', 'slider_diffC_response']].mean(axis=1, skipna=True)
    trials['slider_diffA_response']= trials['slider_diffA_response']
    trials['slider_diffB_response']= trials['slider_diffB_response']
    trials['slider_diffC_response']= trials['slider_diffC_response']
    

# SWITCHING TO CSV

    C1 = [1,4,7]
    C2 = [2,5,8]
    C3 = [3,6,9]
    
    #empty lists for number of switches to append to trials df
    B=[]
    WC1=[]
    WC2=[]
    WC3=[]
    
    first_switch=[]
    first_look =[]
    global gazedata_alltrials
    gazedata_alltrials =[]
    
    #loop for each trial seperate eye data
    for index, trial in trials.iterrows():
        # select eyetracking data for the trial:      
        gazedata = select_eyedata(trial,index,etData)
        gazedata_alltrials.append(gazedata)
        roi = pd.DataFrame(gazedata,columns=['roi']).reset_index(drop=True)
        roi = roi.dropna().reset_index(drop=True)
        if len(roi.index) >= 1:
            first_look.append(roi.roi[0])
        else:
            first_look.append('1')
        
        prev_roi = roi.shift()
        roi['Change'] = np.select([(roi == prev_roi) | prev_roi.isnull(),
        roi.isin(C1) & prev_roi.isin(C1),
        roi.isin(C2) & prev_roi.isin(C2),
        roi.isin(C3) & prev_roi.isin(C3),
        roi != prev_roi],[ "nan", "C1", "C2", "C3", "B",],)
    
        #new dataframe of both
        switch = pd.DataFrame(columns=['subject','Change'])
        switch['Change'] =  roi['Change']
        switch = switch[switch['Change'].str.contains("nan")==False]
        
        if len(switch.index) >= 1:
            first_switch.append((switch['Change'].iloc[0]))
        else:
            first_switch.append('B')    


        B.append((switch['Change'] == 'B').sum())
        WC1.append((switch['Change'] == 'C1').sum())
        WC2.append((switch['Change'] == 'C2').sum())
        WC3.append((switch['Change'] == 'C3').sum())
        
        
    trials ['first_look']= first_look
    trials['first_switch']= first_switch  

    trials['WC1']= WC1
    trials['WC2']= WC2
    trials['WC3']= WC3
    trials['B']= B
    
    trials['W'] = trials['WC1']+trials['WC2']+trials['WC3']
    trials['W_B_total'] =  trials['W'] +  trials['B']
    trials['W'] = trials['W']/trials['W_B_total']
    trials['B'] = trials['B']/trials['W_B_total']
    
    trials['WC1']=trials['WC1']/trials['W']
    trials['WC2']=trials['WC2']/trials['W']
    trials['WC3']=trials['WC3']/trials['W']
    

#SLIDER RATINGS AND DECISONS
   
    #get name of 1,2,3 category based on decision-time for each condition
    trials.loc[(trials['Condition']== 3),'decision1_category'] = trials[['decisiontimeA','decisiontimeB','decisiontimeC']].idxmin(axis = 1)
    trials.loc[(trials['Condition']== 3),'decision2_category'] = trials[['decisiontimeA','decisiontimeB','decisiontimeC']].T.apply(lambda x: x.nlargest(2).idxmin())
    trials.loc[(trials['Condition']== 3),'decision3_category'] = trials[['decisiontimeA','decisiontimeB','decisiontimeC']].idxmax(axis = 1)
      
    trials.loc[trials['decision1_category']=='decisiontimeA','decision1_category'] = 'A'
    trials.loc[trials['decision1_category']=='decisiontimeB','decision1_category'] = 'B'
    trials.loc[trials['decision1_category']=='decisiontimeC','decision1_category'] = 'C'
    
    trials.loc[trials['decision2_category']=='decisiontimeA','decision2_category'] = 'A'
    trials.loc[trials['decision2_category']=='decisiontimeB','decision2_category'] = 'B'
    trials.loc[trials['decision2_category']=='decisiontimeC','decision2_category'] = 'C'
    
    trials.loc[trials['decision3_category']=='decisiontimeA','decision3_category'] = 'A'
    trials.loc[trials['decision3_category']=='decisiontimeB','decision3_category'] = 'B'
    trials.loc[trials['decision3_category']=='decisiontimeC','decision3_category'] = 'C'
    
    #get importance value for 1,2,3 choosen category
    trials.loc[trials['decision1_category']=='A','category1_imp'] = trials['slider_valueA_response']
    trials.loc[trials['decision1_category']=='B','category1_imp'] = trials['slider_valueB_response']
    trials.loc[trials['decision1_category']=='C','category1_imp'] = trials['slider_valueC_response']
    
    trials.loc[trials['decision2_category']=='A','category2_imp'] = trials['slider_valueA_response']
    trials.loc[trials['decision2_category']=='B','category2_imp'] = trials['slider_valueB_response']
    trials.loc[trials['decision2_category']=='C','category2_imp'] = trials['slider_valueC_response']
    
    trials.loc[trials['decision3_category']=='A','category3_imp'] = trials['slider_valueA_response']
    trials.loc[trials['decision3_category']=='B','category3_imp'] = trials['slider_valueB_response']
    trials.loc[trials['decision3_category']=='C','category3_imp'] = trials['slider_valueC_response']
    
    # #get difficulty value for 1,2,3 choosen category
    trials.loc[trials['decision1_category']=='A','category1_diff'] = trials['slider_diffA_response']
    trials.loc[trials['decision1_category']=='B','category1_diff'] = trials['slider_diffB_response']   
    trials.loc[trials['decision1_category']=='C','category1_diff'] = trials['slider_diffC_response'] 
    
    trials.loc[trials['decision2_category']=='A','category2_diff'] = trials['slider_diffA_response']
    trials.loc[trials['decision2_category']=='B','category2_diff'] = trials['slider_diffB_response']   
    trials.loc[trials['decision2_category']=='C','category2_diff'] = trials['slider_diffC_response']   
    
    trials.loc[trials['decision3_category']=='A','category3_diff'] = trials['slider_diffA_response']
    trials.loc[trials['decision3_category']=='B','category3_diff'] = trials['slider_diffB_response']   
    trials.loc[trials['decision3_category']=='C','category3_diff'] = trials['slider_diffC_response']   

    #drop columns which are not needed anymore
    trials = trials.drop(labels = ['t_categoryA','t_categoryB','t_categoryC','n_A1','n_A2','n_A3','n_B1','n_B2','n_B3','n_C1','n_C2','n_C3','diff_mean_cat','imp_mean_cat','slider_valueA_response','slider_valueB_response','slider_valueC_response','slider_diffA_response','slider_diffB_response','slider_diffC_response'], axis = 1).reset_index(drop=True)  
    
    #get looking time for 1,2,3 choosen category
    trials.loc[trials['decision1_category']=='A','f_category1'] = trials['f_categoryA']
    trials.loc[trials['decision1_category']=='B','f_category1'] = trials['f_categoryB']   
    trials.loc[trials['decision1_category']=='C','f_category1'] = trials['f_categoryC'] 
    
    trials.loc[trials['decision2_category']=='A','f_category2'] = trials['f_categoryA']
    trials.loc[trials['decision2_category']=='B','f_category2'] = trials['f_categoryB']   
    trials.loc[trials['decision2_category']=='C','f_category2'] = trials['f_categoryC'] 
    
    trials.loc[trials['decision3_category']=='A','f_category3'] = trials['f_categoryA']
    trials.loc[trials['decision3_category']=='B','f_category3'] = trials['f_categoryB']   
    trials.loc[trials['decision3_category']=='C','f_category3'] = trials['f_categoryC'] 
    
#TRANSLATIONFOR GAZE DATA PLOTS     
    V1 = trials[['decision1_category','decision2_category','decision3_category','clicked_optionA','clicked_optionB','clicked_optionC','A1pos','A2pos','A3pos','B1pos','B2pos','B3pos','C1pos','C2pos','C3pos']] 
        
    V1.loc[trials['decision1_category']=='A','decision1_1category'] = V1['clicked_optionA']
    V1.loc[trials['decision1_category']=='B','decision1_1category'] = V1['clicked_optionB']
    V1.loc[trials['decision1_category']=='C','decision1_1category'] = V1['clicked_optionC']
    
    V1.loc[trials['decision2_category']=='A','decision2_1category'] = V1['clicked_optionA']
    V1.loc[trials['decision2_category']=='B','decision2_1category'] = V1['clicked_optionB']
    V1.loc[trials['decision2_category']=='C','decision2_1category'] = V1['clicked_optionC']
    
    V1.loc[trials['decision3_category']=='A','decision3_1category'] = V1['clicked_optionA']
    V1.loc[trials['decision3_category']=='B','decision3_1category'] = V1['clicked_optionB']
    V1.loc[trials['decision3_category']=='C','decision3_1category'] = V1['clicked_optionC']
    
    V1 = V1.drop(labels = ['clicked_optionA','clicked_optionB','clicked_optionC','decision1_category','decision2_category','decision3_category'], axis = 1).reset_index(drop=True)
   
    V1.A1pos = V1.A1pos.replace({'3_':''}, regex=True)
    V1.A2pos = V1.A2pos.replace({'3_':''}, regex=True)
    V1.A3pos = V1.A3pos.replace({'3_':''}, regex=True)
    V1.B1pos = V1.B1pos.replace({'3_':''}, regex=True)
    V1.B2pos = V1.B2pos.replace({'3_':''}, regex=True)
    V1.B3pos = V1.B3pos.replace({'3_':''}, regex=True)
    V1.C1pos = V1.C1pos.replace({'3_':''}, regex=True)
    V1.C2pos = V1.C2pos.replace({'3_':''}, regex=True)
    V1.C3pos = V1.C3pos.replace({'3_':''}, regex=True)
    
    cols = ['decision1_1category', 'decision2_1category', 'decision3_1category']
    V1[cols] = V1[cols].applymap(lambda lst: eval(lst)[0])
    
    V1.loc[V1['decision1_1category']=='A1','decision1_2category'] = 'A2'
    V1.loc[V1['decision1_1category']=='A1','decision1_3category'] = 'A3'
    
    V1.loc[V1['decision1_1category']=='A2','decision1_2category'] = 'A1'
    V1.loc[V1['decision1_1category']=='A2','decision1_3category'] = 'A3'
    
    V1.loc[V1['decision1_1category']=='A3','decision1_2category'] = 'A1'
    V1.loc[V1['decision1_1category']=='A3','decision1_3category'] = 'A2'
    
    
    V1.loc[V1['decision1_1category']=='B1','decision1_2category'] = 'B2'
    V1.loc[V1['decision1_1category']=='B1','decision1_3category'] = 'B3'
    
    V1.loc[V1['decision1_1category']=='B2','decision1_2category'] = 'B1'
    V1.loc[V1['decision1_1category']=='B2','decision1_3category'] = 'B3'
    
    V1.loc[V1['decision1_1category']=='B3','decision1_2category'] = 'B1'
    V1.loc[V1['decision1_1category']=='B3','decision1_3category'] = 'B2'
    
    
    V1.loc[V1['decision1_1category']=='C1','decision1_2category'] = 'C2'
    V1.loc[V1['decision1_1category']=='C1','decision1_3category'] = 'C3'
    
    V1.loc[V1['decision1_1category']=='C2','decision1_2category'] = 'C1'
    V1.loc[V1['decision1_1category']=='C2','decision1_3category'] = 'C3'
    
    V1.loc[V1['decision1_1category']=='C3','decision1_2category'] = 'C1'
    V1.loc[V1['decision1_1category']=='C3','decision1_3category'] = 'C2'
    
    
    
    V1.loc[V1['decision2_1category']=='A1','decision2_2category'] = 'A2'
    V1.loc[V1['decision2_1category']=='A1','decision2_3category'] = 'A3'
    
    V1.loc[V1['decision2_1category']=='A2','decision2_2category'] = 'A1'
    V1.loc[V1['decision2_1category']=='A2','decision2_3category'] = 'A3'
    
    V1.loc[V1['decision2_1category']=='A3','decision2_2category'] = 'A1'
    V1.loc[V1['decision2_1category']=='A3','decision2_3category'] = 'A2'
    
    
    V1.loc[V1['decision2_1category']=='B1','decision2_2category'] = 'B2'
    V1.loc[V1['decision2_1category']=='B1','decision2_3category'] = 'B3'
    
    V1.loc[V1['decision2_1category']=='B2','decision2_2category'] = 'B1'
    V1.loc[V1['decision2_1category']=='B2','decision2_3category'] = 'B3'
    
    V1.loc[V1['decision2_1category']=='B3','decision2_2category'] = 'B1'
    V1.loc[V1['decision2_1category']=='B3','decision2_3category'] = 'B2'
    
    
    V1.loc[V1['decision2_1category']=='C1','decision2_2category'] = 'C2'
    V1.loc[V1['decision2_1category']=='C1','decision2_3category'] = 'C3'
    
    V1.loc[V1['decision2_1category']=='C2','decision2_2category'] = 'C1'
    V1.loc[V1['decision2_1category']=='C2','decision2_3category'] = 'C3'
    
    V1.loc[V1['decision2_1category']=='C3','decision2_2category'] = 'C1'
    V1.loc[V1['decision2_1category']=='C3','decision2_3category'] = 'C2'



    V1.loc[V1['decision3_1category']=='A1','decision3_2category'] = 'A2'
    V1.loc[V1['decision3_1category']=='A1','decision3_3category'] = 'A3'
    
    V1.loc[V1['decision3_1category']=='A2','decision3_2category'] = 'A1'
    V1.loc[V1['decision3_1category']=='A2','decision3_3category'] = 'A3'
    
    V1.loc[V1['decision3_1category']=='A3','decision3_2category'] = 'A1'
    V1.loc[V1['decision3_1category']=='A3','decision3_3category'] = 'A2'

    
    V1.loc[V1['decision3_1category']=='B1','decision3_2category'] = 'B2'
    V1.loc[V1['decision3_1category']=='B1','decision3_3category'] = 'B3'
    
    V1.loc[V1['decision3_1category']=='B2','decision3_2category'] = 'B1'
    V1.loc[V1['decision3_1category']=='B2','decision3_3category'] = 'B3'
    
    V1.loc[V1['decision3_1category']=='B3','decision3_2category'] = 'B1'
    V1.loc[V1['decision3_1category']=='B3','decision3_3category'] = 'B2'
    
    
    V1.loc[V1['decision3_1category']=='C1','decision3_2category'] = 'C2'
    V1.loc[V1['decision3_1category']=='C1','decision3_3category'] = 'C3'
    
    V1.loc[V1['decision3_1category']=='C2','decision3_2category'] = 'C1'
    V1.loc[V1['decision3_1category']=='C2','decision3_3category'] = 'C3'
    
    V1.loc[V1['decision3_1category']=='C3','decision3_2category'] = 'C1'
    V1.loc[V1['decision3_1category']=='C3','decision3_3category'] = 'C2'
    
    x = ['decision1_1category','decision1_2category','decision1_3category','decision2_1category','decision2_2category','decision2_3category','decision3_1category','decision3_2category','decision3_3category']
     
    for i in x:
        V1.loc[V1[i]=='A1',i] = V1['A1pos']
        V1.loc[V1[i]=='A2',i] = V1['A2pos']
        V1.loc[V1[i]=='A3',i] = V1['A3pos']
        
        V1.loc[V1[i]=='B1',i] = V1['B1pos']
        V1.loc[V1[i]=='B2',i] = V1['B2pos']
        V1.loc[V1[i]=='B3',i] = V1['B3pos']
        
        V1.loc[V1[i]=='C1',i] = V1['C1pos']
        V1.loc[V1[i]=='C2',i] = V1['C2pos']
        V1.loc[V1[i]=='C3',i] = V1['C3pos']
  
    V1 = V1.drop(labels = ['A1pos','A2pos','A3pos','B1pos','B2pos','B3pos','C1pos','C2pos','C3pos'], axis = 1).reset_index(drop=True)
        
    trials['decision1_1category'] = V1['decision1_1category']
    trials['decision1_2category'] = V1['decision1_2category']
    trials['decision1_3category'] = V1['decision1_3category']
    
    trials['decision2_1category'] = V1['decision2_1category']
    trials['decision2_2category'] = V1['decision2_2category']
    trials['decision2_3category'] = V1['decision2_3category']
    
    trials['decision3_1category'] = V1['decision3_1category']
    trials['decision3_2category'] = V1['decision3_2category']
    trials['decision3_3category'] = V1['decision3_3category']
    
    #Add mean values for participant to results frame
    results.loc[results['participant']==trials['participant'].mean(),'validgaze_percent'] = trials['valid_datapoints'].mean()
    
    results_0.loc[results['participant']==trials['participant'].mean(),'W'] = trials['W'].mean()
    results_0.loc[results['participant']==trials['participant'].mean(),'B'] = trials['B'].mean()
    
    results_0.loc[results['participant']==trials['participant'].mean(),'f_category1'] = trials['f_category1'].mean()
    results_0.loc[results['participant']==trials['participant'].mean(),'f_category2'] = trials['f_category2'].mean()
    results_0.loc[results['participant']==trials['participant'].mean(),'f_category3'] = trials['f_category3'].mean()

    
    results_0.loc[results['participant']==trials['participant'].mean(),'decisiontime1'] = trials['decisiontime1'].mean()
    results_0.loc[results['participant']==trials['participant'].mean(),'decisiontime2'] = trials['decisiontime2'].mean()
    results_0.loc[results['participant']==trials['participant'].mean(),'decisiontimetotal'] = trials['decisiontimetotal'].mean()
    
    results_0.loc[results['participant']==trials['participant'].mean(),'category1_imp'] = trials['category1_imp'].mean()
    results_0.loc[results['participant']==trials['participant'].mean(),'category2_imp'] = trials['category2_imp'].mean()
    results_0.loc[results['participant']==trials['participant'].mean(),'category3_imp'] = trials['category3_imp'].mean()
    
    results_0.loc[results['participant']==trials['participant'].mean(),'category1_diff'] = trials['category1_diff'].mean()
    results_0.loc[results['participant']==trials['participant'].mean(),'category2_diff'] = trials['category2_diff'].mean()
    results_0.loc[results['participant']==trials['participant'].mean(),'category3_diff'] = trials['category3_diff'].mean()
    
    results_0.loc[results['participant']==trials['participant'].mean(),'WC1'] = trials['WC1'].mean()
    results_0.loc[results['participant']==trials['participant'].mean(),'WC2'] = trials['WC2'].mean()
    results_0.loc[results['participant']==trials['participant'].mean(),'WC3'] = trials['WC3'].mean()  

def prepare_csv_1():
    global trials
    
#LOOKING TIME TO CSV
    list3_1=[]
    list3_2=[]
    list3_3=[]
    list3_4=[]
    list3_5=[]
    list3_6=[]
    list3_7=[]
    list3_8=[]
    list3_9=[]
    invalid_trials =[]

    #loop for each trial seperate eye data
    for index, trial in trials.iterrows():
        # select eyetracking data for the trial:      
        gazedata = select_eyedata(trial,index,etData)
        invalid_trials.append(invalid_datapoints[::2])
        
        #calculate eye data point sum for each region of interest and trial
        list3_1.append(gazedata['3_1'].sum())
        list3_2.append(gazedata['3_2'].sum())
        list3_3.append(gazedata['3_3'].sum())
        list3_4.append(gazedata['3_4'].sum())
        list3_5.append(gazedata['3_5'].sum())
        list3_6.append(gazedata['3_6'].sum())
        list3_7.append(gazedata['3_7'].sum())
        list3_8.append(gazedata['3_8'].sum())
        list3_9.append(gazedata['3_9'].sum())
    
    #get valid datapoint proportion for each trial
    flat_invalid_trials = [item for sublist in invalid_trials for item in sublist]
    trials['valid_datapoints'] = flat_invalid_trials  
    trials['valid_datapoints'] =(trials['valid_datapoints'])*100
    #trials = trials.drop(trials[trials['valid_trial']<0.25].index).reset_index(drop=True)
    

    # add number of region of interest gaze points to csv per trial
    trials['n_3_1']= list3_1
    trials['n_3_2']= list3_2
    trials['n_3_3']= list3_3
    trials['n_3_4']= list3_4
    trials['n_3_5']= list3_5
    trials['n_3_6']= list3_6
    trials['n_3_7']= list3_7
    trials['n_3_8']= list3_8
    trials['n_3_9']= list3_9
    
    # use location eye data to realte it to image looking time 
    start = [3]
    end_3 = [1,2,3,4,5,6,7,8,9]

    
    for start_n in start:
         if start_n == 3:
             end = end_3
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['A1pos']==str(start_n)+'_'+str(end_n),['n_A1']] = trials.loc[trials['A1pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['A2pos']==str(start_n)+'_'+str(end_n),['n_A2']] = trials.loc[trials['A2pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['A3pos']==str(start_n)+'_'+str(end_n),['n_A3']] = trials.loc[trials['A3pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['B1pos']==str(start_n)+'_'+str(end_n),['n_B1']] = trials.loc[trials['B1pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values           
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['B2pos']==str(start_n)+'_'+str(end_n),['n_B2']] = trials.loc[trials['B2pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['B3pos']==str(start_n)+'_'+str(end_n),['n_B3']] = trials.loc[trials['B3pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['C1pos']==str(start_n)+'_'+str(end_n),['n_C1']] = trials.loc[trials['C1pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['C2pos']==str(start_n)+'_'+str(end_n),['n_C2']] = trials.loc[trials['C2pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values           
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['C3pos']==str(start_n)+'_'+str(end_n),['n_C3']] = trials.loc[trials['C3pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values               
                    
    # change data points to time 
    trials['t_A1'] = trials['n_A1']/120
    trials['t_A2'] = trials['n_A2']/120
    trials['t_A3'] = trials['n_A3']/120
    trials['t_B1'] = trials['n_B1']/120
    trials['t_B2'] = trials['n_B2']/120
    trials['t_B3'] = trials['n_B3']/120
    trials['t_C1'] = trials['n_C1']/120
    trials['t_C2'] = trials['n_C2']/120
    trials['t_C3'] = trials['n_C3']/120       

    trials.loc[(trials['Condition']== 3),'t_categoryA'] = trials['t_A1']+ trials['t_A2']+ trials['t_A3']
    trials.loc[(trials['Condition']== 3),'t_categoryB'] = trials['t_B1']+ trials['t_B2']+ trials['t_B3']
    trials.loc[(trials['Condition']== 3),'t_categoryC'] = trials['t_C1']+ trials['t_C2']+ trials['t_C3']

   #get fraction of lookin time per category
    trials.loc[(trials['Condition']== 3),'t_cat_total'] = trials['t_categoryA']+ trials['t_categoryB']+ trials['t_categoryC']
   
    trials['f_categoryA']= trials['t_categoryA']/trials['t_cat_total']
    trials['f_categoryB']= trials['t_categoryB']/trials['t_cat_total']
    trials['f_categoryC']= trials['t_categoryC']/trials['t_cat_total']
    
    #change slider values to standart ones 
    trials.loc[(trials['Condition']== 3),'imp_mean_cat'] = trials[['slider_valueA_response', 'slider_valueB_response', 'slider_valueC_response']].mean(axis=1, skipna=True)
    trials['slider_valueA_response']= trials['slider_valueA_response']
    trials['slider_valueB_response']= trials['slider_valueB_response']
    trials['slider_valueC_response']= trials['slider_valueC_response']
    
    trials.loc[(trials['Condition']== 3),'diff_mean_cat'] = trials[['slider_diffA_response', 'slider_diffB_response', 'slider_diffC_response']].mean(axis=1, skipna=True)
    trials['slider_diffA_response']= trials['slider_diffA_response']
    trials['slider_diffB_response']= trials['slider_diffB_response']
    trials['slider_diffC_response']= trials['slider_diffC_response']
    

# SWITCHING TO CSV

    C1 = [1,4,7]
    C2 = [2,5,8]
    C3 = [3,6,9]
    
    #empty lists for number of switches to append to trials df
    B=[]
    WC1=[]
    WC2=[]
    WC3=[]
    
    first_switch=[]
    first_look =[]
    global gazedata_alltrials
    gazedata_alltrials =[]
    
    #loop for each trial seperate eye data
    for index, trial in trials.iterrows():
        # select eyetracking data for the trial:      
        gazedata = select_eyedata(trial,index,etData)
        gazedata_alltrials.append(gazedata)
        roi = pd.DataFrame(gazedata,columns=['roi']).reset_index(drop=True)
        roi = roi.dropna().reset_index(drop=True)
        if len(roi.index) >= 1:
            first_look.append(roi.roi[0])
        else:
            first_look.append('1')
        
        prev_roi = roi.shift()
        roi['Change'] = np.select([(roi == prev_roi) | prev_roi.isnull(),
        roi.isin(C1) & prev_roi.isin(C1),
        roi.isin(C2) & prev_roi.isin(C2),
        roi.isin(C3) & prev_roi.isin(C3),
        roi != prev_roi],[ "nan", "C1", "C2", "C3", "B",],)
    
        #new dataframe of both
        switch = pd.DataFrame(columns=['subject','Change'])
        switch['Change'] =  roi['Change']
        switch = switch[switch['Change'].str.contains("nan")==False]
        
        if len(switch.index) >= 1:
            first_switch.append((switch['Change'].iloc[0]))
        else:
            first_switch.append('B')    


        B.append((switch['Change'] == 'B').sum())
        WC1.append((switch['Change'] == 'C1').sum())
        WC2.append((switch['Change'] == 'C2').sum())
        WC3.append((switch['Change'] == 'C3').sum())
        
        
    trials ['first_look']= first_look
    trials['first_switch']= first_switch  

    trials['WC1']= WC1
    trials['WC2']= WC2
    trials['WC3']= WC3
    trials['B']= B
    
    trials['W'] = trials['WC1']+trials['WC2']+trials['WC3']
    trials['W_B_total'] =  trials['W'] +  trials['B']
    trials['W'] = trials['W']/trials['W_B_total']
    trials['B'] = trials['B']/trials['W_B_total']
    
    trials['WC1']=trials['WC1']/trials['W']
    trials['WC2']=trials['WC2']/trials['W']
    trials['WC3']=trials['WC3']/trials['W']
    

#SLIDER RATINGS AND DECISONS
   
    #get name of 1,2,3 category based on decision-time for each condition
    trials.loc[(trials['Condition']== 3),'decision1_category'] = trials[['decisiontimeA','decisiontimeB','decisiontimeC']].idxmin(axis = 1)
    trials.loc[(trials['Condition']== 3),'decision2_category'] = trials[['decisiontimeA','decisiontimeB','decisiontimeC']].T.apply(lambda x: x.nlargest(2).idxmin())
    trials.loc[(trials['Condition']== 3),'decision3_category'] = trials[['decisiontimeA','decisiontimeB','decisiontimeC']].idxmax(axis = 1)
      
    trials.loc[trials['decision1_category']=='decisiontimeA','decision1_category'] = 'A'
    trials.loc[trials['decision1_category']=='decisiontimeB','decision1_category'] = 'B'
    trials.loc[trials['decision1_category']=='decisiontimeC','decision1_category'] = 'C'
    
    trials.loc[trials['decision2_category']=='decisiontimeA','decision2_category'] = 'A'
    trials.loc[trials['decision2_category']=='decisiontimeB','decision2_category'] = 'B'
    trials.loc[trials['decision2_category']=='decisiontimeC','decision2_category'] = 'C'
    
    trials.loc[trials['decision3_category']=='decisiontimeA','decision3_category'] = 'A'
    trials.loc[trials['decision3_category']=='decisiontimeB','decision3_category'] = 'B'
    trials.loc[trials['decision3_category']=='decisiontimeC','decision3_category'] = 'C'
    
    #get importance value for 1,2,3 choosen category
    trials.loc[trials['decision1_category']=='A','category1_imp'] = trials['slider_valueA_response']
    trials.loc[trials['decision1_category']=='B','category1_imp'] = trials['slider_valueB_response']
    trials.loc[trials['decision1_category']=='C','category1_imp'] = trials['slider_valueC_response']
    
    trials.loc[trials['decision2_category']=='A','category2_imp'] = trials['slider_valueA_response']
    trials.loc[trials['decision2_category']=='B','category2_imp'] = trials['slider_valueB_response']
    trials.loc[trials['decision2_category']=='C','category2_imp'] = trials['slider_valueC_response']
    
    trials.loc[trials['decision3_category']=='A','category3_imp'] = trials['slider_valueA_response']
    trials.loc[trials['decision3_category']=='B','category3_imp'] = trials['slider_valueB_response']
    trials.loc[trials['decision3_category']=='C','category3_imp'] = trials['slider_valueC_response']
    
    # #get difficulty value for 1,2,3 choosen category
    trials.loc[trials['decision1_category']=='A','category1_diff'] = trials['slider_diffA_response']
    trials.loc[trials['decision1_category']=='B','category1_diff'] = trials['slider_diffB_response']   
    trials.loc[trials['decision1_category']=='C','category1_diff'] = trials['slider_diffC_response'] 
    
    trials.loc[trials['decision2_category']=='A','category2_diff'] = trials['slider_diffA_response']
    trials.loc[trials['decision2_category']=='B','category2_diff'] = trials['slider_diffB_response']   
    trials.loc[trials['decision2_category']=='C','category2_diff'] = trials['slider_diffC_response']   
    
    trials.loc[trials['decision3_category']=='A','category3_diff'] = trials['slider_diffA_response']
    trials.loc[trials['decision3_category']=='B','category3_diff'] = trials['slider_diffB_response']   
    trials.loc[trials['decision3_category']=='C','category3_diff'] = trials['slider_diffC_response']   

    #drop columns which are not needed anymore
    trials = trials.drop(labels = ['t_categoryA','t_categoryB','t_categoryC','n_A1','n_A2','n_A3','n_B1','n_B2','n_B3','n_C1','n_C2','n_C3','diff_mean_cat','imp_mean_cat','slider_valueA_response','slider_valueB_response','slider_valueC_response','slider_diffA_response','slider_diffB_response','slider_diffC_response'], axis = 1).reset_index(drop=True)  
    
    #get looking time for 1,2,3 choosen category
    trials.loc[trials['decision1_category']=='A','f_category1'] = trials['f_categoryA']
    trials.loc[trials['decision1_category']=='B','f_category1'] = trials['f_categoryB']   
    trials.loc[trials['decision1_category']=='C','f_category1'] = trials['f_categoryC'] 
    
    trials.loc[trials['decision2_category']=='A','f_category2'] = trials['f_categoryA']
    trials.loc[trials['decision2_category']=='B','f_category2'] = trials['f_categoryB']   
    trials.loc[trials['decision2_category']=='C','f_category2'] = trials['f_categoryC'] 
    
    trials.loc[trials['decision3_category']=='A','f_category3'] = trials['f_categoryA']
    trials.loc[trials['decision3_category']=='B','f_category3'] = trials['f_categoryB']   
    trials.loc[trials['decision3_category']=='C','f_category3'] = trials['f_categoryC'] 
    
#TRANSLATIONFOR GAZE DATA PLOTS     
    V1 = trials[['decision1_category','decision2_category','decision3_category','clicked_optionA','clicked_optionB','clicked_optionC','A1pos','A2pos','A3pos','B1pos','B2pos','B3pos','C1pos','C2pos','C3pos']] 
        
    V1.loc[trials['decision1_category']=='A','decision1_1category'] = V1['clicked_optionA']
    V1.loc[trials['decision1_category']=='B','decision1_1category'] = V1['clicked_optionB']
    V1.loc[trials['decision1_category']=='C','decision1_1category'] = V1['clicked_optionC']
    
    V1.loc[trials['decision2_category']=='A','decision2_1category'] = V1['clicked_optionA']
    V1.loc[trials['decision2_category']=='B','decision2_1category'] = V1['clicked_optionB']
    V1.loc[trials['decision2_category']=='C','decision2_1category'] = V1['clicked_optionC']
    
    V1.loc[trials['decision3_category']=='A','decision3_1category'] = V1['clicked_optionA']
    V1.loc[trials['decision3_category']=='B','decision3_1category'] = V1['clicked_optionB']
    V1.loc[trials['decision3_category']=='C','decision3_1category'] = V1['clicked_optionC']
    
    V1 = V1.drop(labels = ['clicked_optionA','clicked_optionB','clicked_optionC','decision1_category','decision2_category','decision3_category'], axis = 1).reset_index(drop=True)
   
    V1.A1pos = V1.A1pos.replace({'3_':''}, regex=True)
    V1.A2pos = V1.A2pos.replace({'3_':''}, regex=True)
    V1.A3pos = V1.A3pos.replace({'3_':''}, regex=True)
    V1.B1pos = V1.B1pos.replace({'3_':''}, regex=True)
    V1.B2pos = V1.B2pos.replace({'3_':''}, regex=True)
    V1.B3pos = V1.B3pos.replace({'3_':''}, regex=True)
    V1.C1pos = V1.C1pos.replace({'3_':''}, regex=True)
    V1.C2pos = V1.C2pos.replace({'3_':''}, regex=True)
    V1.C3pos = V1.C3pos.replace({'3_':''}, regex=True)
    
    cols = ['decision1_1category', 'decision2_1category', 'decision3_1category']
    V1[cols] = V1[cols].applymap(lambda lst: eval(lst)[0])
    
    V1.loc[V1['decision1_1category']=='A1','decision1_2category'] = 'A2'
    V1.loc[V1['decision1_1category']=='A1','decision1_3category'] = 'A3'
    
    V1.loc[V1['decision1_1category']=='A2','decision1_2category'] = 'A1'
    V1.loc[V1['decision1_1category']=='A2','decision1_3category'] = 'A3'
    
    V1.loc[V1['decision1_1category']=='A3','decision1_2category'] = 'A1'
    V1.loc[V1['decision1_1category']=='A3','decision1_3category'] = 'A2'
    
    
    V1.loc[V1['decision1_1category']=='B1','decision1_2category'] = 'B2'
    V1.loc[V1['decision1_1category']=='B1','decision1_3category'] = 'B3'
    
    V1.loc[V1['decision1_1category']=='B2','decision1_2category'] = 'B1'
    V1.loc[V1['decision1_1category']=='B2','decision1_3category'] = 'B3'
    
    V1.loc[V1['decision1_1category']=='B3','decision1_2category'] = 'B1'
    V1.loc[V1['decision1_1category']=='B3','decision1_3category'] = 'B2'
    
    
    V1.loc[V1['decision1_1category']=='C1','decision1_2category'] = 'C2'
    V1.loc[V1['decision1_1category']=='C1','decision1_3category'] = 'C3'
    
    V1.loc[V1['decision1_1category']=='C2','decision1_2category'] = 'C1'
    V1.loc[V1['decision1_1category']=='C2','decision1_3category'] = 'C3'
    
    V1.loc[V1['decision1_1category']=='C3','decision1_2category'] = 'C1'
    V1.loc[V1['decision1_1category']=='C3','decision1_3category'] = 'C2'
    
    
    
    V1.loc[V1['decision2_1category']=='A1','decision2_2category'] = 'A2'
    V1.loc[V1['decision2_1category']=='A1','decision2_3category'] = 'A3'
    
    V1.loc[V1['decision2_1category']=='A2','decision2_2category'] = 'A1'
    V1.loc[V1['decision2_1category']=='A2','decision2_3category'] = 'A3'
    
    V1.loc[V1['decision2_1category']=='A3','decision2_2category'] = 'A1'
    V1.loc[V1['decision2_1category']=='A3','decision2_3category'] = 'A2'
    
    
    V1.loc[V1['decision2_1category']=='B1','decision2_2category'] = 'B2'
    V1.loc[V1['decision2_1category']=='B1','decision2_3category'] = 'B3'
    
    V1.loc[V1['decision2_1category']=='B2','decision2_2category'] = 'B1'
    V1.loc[V1['decision2_1category']=='B2','decision2_3category'] = 'B3'
    
    V1.loc[V1['decision2_1category']=='B3','decision2_2category'] = 'B1'
    V1.loc[V1['decision2_1category']=='B3','decision2_3category'] = 'B2'
    
    
    V1.loc[V1['decision2_1category']=='C1','decision2_2category'] = 'C2'
    V1.loc[V1['decision2_1category']=='C1','decision2_3category'] = 'C3'
    
    V1.loc[V1['decision2_1category']=='C2','decision2_2category'] = 'C1'
    V1.loc[V1['decision2_1category']=='C2','decision2_3category'] = 'C3'
    
    V1.loc[V1['decision2_1category']=='C3','decision2_2category'] = 'C1'
    V1.loc[V1['decision2_1category']=='C3','decision2_3category'] = 'C2'



    V1.loc[V1['decision3_1category']=='A1','decision3_2category'] = 'A2'
    V1.loc[V1['decision3_1category']=='A1','decision3_3category'] = 'A3'
    
    V1.loc[V1['decision3_1category']=='A2','decision3_2category'] = 'A1'
    V1.loc[V1['decision3_1category']=='A2','decision3_3category'] = 'A3'
    
    V1.loc[V1['decision3_1category']=='A3','decision3_2category'] = 'A1'
    V1.loc[V1['decision3_1category']=='A3','decision3_3category'] = 'A2'

    
    V1.loc[V1['decision3_1category']=='B1','decision3_2category'] = 'B2'
    V1.loc[V1['decision3_1category']=='B1','decision3_3category'] = 'B3'
    
    V1.loc[V1['decision3_1category']=='B2','decision3_2category'] = 'B1'
    V1.loc[V1['decision3_1category']=='B2','decision3_3category'] = 'B3'
    
    V1.loc[V1['decision3_1category']=='B3','decision3_2category'] = 'B1'
    V1.loc[V1['decision3_1category']=='B3','decision3_3category'] = 'B2'
    
    
    V1.loc[V1['decision3_1category']=='C1','decision3_2category'] = 'C2'
    V1.loc[V1['decision3_1category']=='C1','decision3_3category'] = 'C3'
    
    V1.loc[V1['decision3_1category']=='C2','decision3_2category'] = 'C1'
    V1.loc[V1['decision3_1category']=='C2','decision3_3category'] = 'C3'
    
    V1.loc[V1['decision3_1category']=='C3','decision3_2category'] = 'C1'
    V1.loc[V1['decision3_1category']=='C3','decision3_3category'] = 'C2'
    
    x = ['decision1_1category','decision1_2category','decision1_3category','decision2_1category','decision2_2category','decision2_3category','decision3_1category','decision3_2category','decision3_3category']
     
    for i in x:
        V1.loc[V1[i]=='A1',i] = V1['A1pos']
        V1.loc[V1[i]=='A2',i] = V1['A2pos']
        V1.loc[V1[i]=='A3',i] = V1['A3pos']
        
        V1.loc[V1[i]=='B1',i] = V1['B1pos']
        V1.loc[V1[i]=='B2',i] = V1['B2pos']
        V1.loc[V1[i]=='B3',i] = V1['B3pos']
        
        V1.loc[V1[i]=='C1',i] = V1['C1pos']
        V1.loc[V1[i]=='C2',i] = V1['C2pos']
        V1.loc[V1[i]=='C3',i] = V1['C3pos']
  
    V1 = V1.drop(labels = ['A1pos','A2pos','A3pos','B1pos','B2pos','B3pos','C1pos','C2pos','C3pos'], axis = 1).reset_index(drop=True)
        
    trials['decision1_1category'] = V1['decision1_1category']
    trials['decision1_2category'] = V1['decision1_2category']
    trials['decision1_3category'] = V1['decision1_3category']
    
    trials['decision2_1category'] = V1['decision2_1category']
    trials['decision2_2category'] = V1['decision2_2category']
    trials['decision2_3category'] = V1['decision2_3category']
    
    trials['decision3_1category'] = V1['decision3_1category']
    trials['decision3_2category'] = V1['decision3_2category']
    trials['decision3_3category'] = V1['decision3_3category']
    
    #Add mean values for participant to results frame
    results.loc[results['participant']==trials['participant'].mean(),'validgaze_percent'] = trials['valid_datapoints'].mean()
    
    results_1.loc[results['participant']==trials['participant'].mean(),'W'] = trials['W'].mean()
    results_1.loc[results['participant']==trials['participant'].mean(),'B'] = trials['B'].mean()
    
    results_1.loc[results['participant']==trials['participant'].mean(),'f_category1'] = trials['f_category1'].mean()
    results_1.loc[results['participant']==trials['participant'].mean(),'f_category2'] = trials['f_category2'].mean()
    results_1.loc[results['participant']==trials['participant'].mean(),'f_category3'] = trials['f_category3'].mean()

    
    results_1.loc[results['participant']==trials['participant'].mean(),'decisiontime1'] = trials['decisiontime1'].mean()
    results_1.loc[results['participant']==trials['participant'].mean(),'decisiontime2'] = trials['decisiontime2'].mean()
    results_1.loc[results['participant']==trials['participant'].mean(),'decisiontimetotal'] = trials['decisiontimetotal'].mean()
    
    results_1.loc[results['participant']==trials['participant'].mean(),'category1_imp'] = trials['category1_imp'].mean()
    results_1.loc[results['participant']==trials['participant'].mean(),'category2_imp'] = trials['category2_imp'].mean()
    results_1.loc[results['participant']==trials['participant'].mean(),'category3_imp'] = trials['category3_imp'].mean()
    
    results_1.loc[results['participant']==trials['participant'].mean(),'category1_diff'] = trials['category1_diff'].mean()
    results_1.loc[results['participant']==trials['participant'].mean(),'category2_diff'] = trials['category2_diff'].mean()
    results_1.loc[results['participant']==trials['participant'].mean(),'category3_diff'] = trials['category3_diff'].mean()
    
    results_1.loc[results['participant']==trials['participant'].mean(),'WC1'] = trials['WC1'].mean()
    results_1.loc[results['participant']==trials['participant'].mean(),'WC2'] = trials['WC2'].mean()
    results_1.loc[results['participant']==trials['participant'].mean(),'WC3'] = trials['WC3'].mean()  
    
def prepare_csv_2():
    global trials
    
#LOOKING TIME TO CSV
    list3_1=[]
    list3_2=[]
    list3_3=[]
    list3_4=[]
    list3_5=[]
    list3_6=[]
    list3_7=[]
    list3_8=[]
    list3_9=[]
    invalid_trials =[]

    #loop for each trial seperate eye data
    for index, trial in trials.iterrows():
        # select eyetracking data for the trial:      
        gazedata = select_eyedata(trial,index,etData)
        invalid_trials.append(invalid_datapoints[::2])
        
        #calculate eye data point sum for each region of interest and trial
        list3_1.append(gazedata['3_1'].sum())
        list3_2.append(gazedata['3_2'].sum())
        list3_3.append(gazedata['3_3'].sum())
        list3_4.append(gazedata['3_4'].sum())
        list3_5.append(gazedata['3_5'].sum())
        list3_6.append(gazedata['3_6'].sum())
        list3_7.append(gazedata['3_7'].sum())
        list3_8.append(gazedata['3_8'].sum())
        list3_9.append(gazedata['3_9'].sum())
    
    #get valid datapoint proportion for each trial
    flat_invalid_trials = [item for sublist in invalid_trials for item in sublist]
    trials['valid_datapoints'] = flat_invalid_trials  
    trials['valid_datapoints'] =(trials['valid_datapoints'])*100
    #trials = trials.drop(trials[trials['valid_trial']<0.25].index).reset_index(drop=True)
    

    # add number of region of interest gaze points to csv per trial
    trials['n_3_1']= list3_1
    trials['n_3_2']= list3_2
    trials['n_3_3']= list3_3
    trials['n_3_4']= list3_4
    trials['n_3_5']= list3_5
    trials['n_3_6']= list3_6
    trials['n_3_7']= list3_7
    trials['n_3_8']= list3_8
    trials['n_3_9']= list3_9
    
    # use location eye data to realte it to image looking time 
    start = [3]
    end_3 = [1,2,3,4,5,6,7,8,9]

    
    for start_n in start:
         if start_n == 3:
             end = end_3
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['A1pos']==str(start_n)+'_'+str(end_n),['n_A1']] = trials.loc[trials['A1pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['A2pos']==str(start_n)+'_'+str(end_n),['n_A2']] = trials.loc[trials['A2pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['A3pos']==str(start_n)+'_'+str(end_n),['n_A3']] = trials.loc[trials['A3pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['B1pos']==str(start_n)+'_'+str(end_n),['n_B1']] = trials.loc[trials['B1pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values           
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['B2pos']==str(start_n)+'_'+str(end_n),['n_B2']] = trials.loc[trials['B2pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['B3pos']==str(start_n)+'_'+str(end_n),['n_B3']] = trials.loc[trials['B3pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['C1pos']==str(start_n)+'_'+str(end_n),['n_C1']] = trials.loc[trials['C1pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['C2pos']==str(start_n)+'_'+str(end_n),['n_C2']] = trials.loc[trials['C2pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values           
         for end_n in end:
             boolean = [str(start_n)+'_'+str(end_n) in column for column in trials.keys()]
             trials.loc[trials['C3pos']==str(start_n)+'_'+str(end_n),['n_C3']] = trials.loc[trials['C3pos']==str(start_n)+'_'+str(end_n),trials.keys()[boolean]].values               
                    
    # change data points to time 
    trials['t_A1'] = trials['n_A1']/120
    trials['t_A2'] = trials['n_A2']/120
    trials['t_A3'] = trials['n_A3']/120
    trials['t_B1'] = trials['n_B1']/120
    trials['t_B2'] = trials['n_B2']/120
    trials['t_B3'] = trials['n_B3']/120
    trials['t_C1'] = trials['n_C1']/120
    trials['t_C2'] = trials['n_C2']/120
    trials['t_C3'] = trials['n_C3']/120       

    trials.loc[(trials['Condition']== 3),'t_categoryA'] = trials['t_A1']+ trials['t_A2']+ trials['t_A3']
    trials.loc[(trials['Condition']== 3),'t_categoryB'] = trials['t_B1']+ trials['t_B2']+ trials['t_B3']
    trials.loc[(trials['Condition']== 3),'t_categoryC'] = trials['t_C1']+ trials['t_C2']+ trials['t_C3']

   #get fraction of lookin time per category
    trials.loc[(trials['Condition']== 3),'t_cat_total'] = trials['t_categoryA']+ trials['t_categoryB']+ trials['t_categoryC']
   
    trials['f_categoryA']= trials['t_categoryA']/trials['t_cat_total']
    trials['f_categoryB']= trials['t_categoryB']/trials['t_cat_total']
    trials['f_categoryC']= trials['t_categoryC']/trials['t_cat_total']
    
    #change slider values to standart ones 
    trials.loc[(trials['Condition']== 3),'imp_mean_cat'] = trials[['slider_valueA_response', 'slider_valueB_response', 'slider_valueC_response']].mean(axis=1, skipna=True)
    trials['slider_valueA_response']= trials['slider_valueA_response']
    trials['slider_valueB_response']= trials['slider_valueB_response']
    trials['slider_valueC_response']= trials['slider_valueC_response']
    
    trials.loc[(trials['Condition']== 3),'diff_mean_cat'] = trials[['slider_diffA_response', 'slider_diffB_response', 'slider_diffC_response']].mean(axis=1, skipna=True)
    trials['slider_diffA_response']= trials['slider_diffA_response']
    trials['slider_diffB_response']= trials['slider_diffB_response']
    trials['slider_diffC_response']= trials['slider_diffC_response']
    

# SWITCHING TO CSV

    C1 = [1,4,7]
    C2 = [2,5,8]
    C3 = [3,6,9]
    
    #empty lists for number of switches to append to trials df
    B=[]
    WC1=[]
    WC2=[]
    WC3=[]
    
    first_switch=[]
    first_look =[]
    global gazedata_alltrials
    gazedata_alltrials =[]
    
    #loop for each trial seperate eye data
    for index, trial in trials.iterrows():
        # select eyetracking data for the trial:      
        gazedata = select_eyedata(trial,index,etData)
        gazedata_alltrials.append(gazedata)
        roi = pd.DataFrame(gazedata,columns=['roi']).reset_index(drop=True)
        roi = roi.dropna().reset_index(drop=True)
        if len(roi.index) >= 1:
            first_look.append(roi.roi[0])
        else:
            first_look.append('1')
        
        prev_roi = roi.shift()
        roi['Change'] = np.select([(roi == prev_roi) | prev_roi.isnull(),
        roi.isin(C1) & prev_roi.isin(C1),
        roi.isin(C2) & prev_roi.isin(C2),
        roi.isin(C3) & prev_roi.isin(C3),
        roi != prev_roi],[ "nan", "C1", "C2", "C3", "B",],)
    
        #new dataframe of both
        switch = pd.DataFrame(columns=['subject','Change'])
        switch['Change'] =  roi['Change']
        switch = switch[switch['Change'].str.contains("nan")==False]
        
        if len(switch.index) >= 1:
            first_switch.append((switch['Change'].iloc[0]))
        else:
            first_switch.append('B')    


        B.append((switch['Change'] == 'B').sum())
        WC1.append((switch['Change'] == 'C1').sum())
        WC2.append((switch['Change'] == 'C2').sum())
        WC3.append((switch['Change'] == 'C3').sum())
        
        
    trials ['first_look']= first_look
    trials['first_switch']= first_switch  

    trials['WC1']= WC1
    trials['WC2']= WC2
    trials['WC3']= WC3
    trials['B']= B
    
    trials['W'] = trials['WC1']+trials['WC2']+trials['WC3']
    trials['W_B_total'] =  trials['W'] +  trials['B']
    trials['W'] = trials['W']/trials['W_B_total']
    trials['B'] = trials['B']/trials['W_B_total']
    
    trials['WC1']=trials['WC1']/trials['W']
    trials['WC2']=trials['WC2']/trials['W']
    trials['WC3']=trials['WC3']/trials['W']
    

#SLIDER RATINGS AND DECISONS
   
    #get name of 1,2,3 category based on decision-time for each condition
    trials.loc[(trials['Condition']== 3),'decision1_category'] = trials[['decisiontimeA','decisiontimeB','decisiontimeC']].idxmin(axis = 1)
    trials.loc[(trials['Condition']== 3),'decision2_category'] = trials[['decisiontimeA','decisiontimeB','decisiontimeC']].T.apply(lambda x: x.nlargest(2).idxmin())
    trials.loc[(trials['Condition']== 3),'decision3_category'] = trials[['decisiontimeA','decisiontimeB','decisiontimeC']].idxmax(axis = 1)
      
    trials.loc[trials['decision1_category']=='decisiontimeA','decision1_category'] = 'A'
    trials.loc[trials['decision1_category']=='decisiontimeB','decision1_category'] = 'B'
    trials.loc[trials['decision1_category']=='decisiontimeC','decision1_category'] = 'C'
    
    trials.loc[trials['decision2_category']=='decisiontimeA','decision2_category'] = 'A'
    trials.loc[trials['decision2_category']=='decisiontimeB','decision2_category'] = 'B'
    trials.loc[trials['decision2_category']=='decisiontimeC','decision2_category'] = 'C'
    
    trials.loc[trials['decision3_category']=='decisiontimeA','decision3_category'] = 'A'
    trials.loc[trials['decision3_category']=='decisiontimeB','decision3_category'] = 'B'
    trials.loc[trials['decision3_category']=='decisiontimeC','decision3_category'] = 'C'
    
    #get importance value for 1,2,3 choosen category
    trials.loc[trials['decision1_category']=='A','category1_imp'] = trials['slider_valueA_response']
    trials.loc[trials['decision1_category']=='B','category1_imp'] = trials['slider_valueB_response']
    trials.loc[trials['decision1_category']=='C','category1_imp'] = trials['slider_valueC_response']
    
    trials.loc[trials['decision2_category']=='A','category2_imp'] = trials['slider_valueA_response']
    trials.loc[trials['decision2_category']=='B','category2_imp'] = trials['slider_valueB_response']
    trials.loc[trials['decision2_category']=='C','category2_imp'] = trials['slider_valueC_response']
    
    trials.loc[trials['decision3_category']=='A','category3_imp'] = trials['slider_valueA_response']
    trials.loc[trials['decision3_category']=='B','category3_imp'] = trials['slider_valueB_response']
    trials.loc[trials['decision3_category']=='C','category3_imp'] = trials['slider_valueC_response']
    
    # #get difficulty value for 1,2,3 choosen category
    trials.loc[trials['decision1_category']=='A','category1_diff'] = trials['slider_diffA_response']
    trials.loc[trials['decision1_category']=='B','category1_diff'] = trials['slider_diffB_response']   
    trials.loc[trials['decision1_category']=='C','category1_diff'] = trials['slider_diffC_response'] 
    
    trials.loc[trials['decision2_category']=='A','category2_diff'] = trials['slider_diffA_response']
    trials.loc[trials['decision2_category']=='B','category2_diff'] = trials['slider_diffB_response']   
    trials.loc[trials['decision2_category']=='C','category2_diff'] = trials['slider_diffC_response']   
    
    trials.loc[trials['decision3_category']=='A','category3_diff'] = trials['slider_diffA_response']
    trials.loc[trials['decision3_category']=='B','category3_diff'] = trials['slider_diffB_response']   
    trials.loc[trials['decision3_category']=='C','category3_diff'] = trials['slider_diffC_response']   

    #drop columns which are not needed anymore
    trials = trials.drop(labels = ['t_categoryA','t_categoryB','t_categoryC','n_A1','n_A2','n_A3','n_B1','n_B2','n_B3','n_C1','n_C2','n_C3','diff_mean_cat','imp_mean_cat','slider_valueA_response','slider_valueB_response','slider_valueC_response','slider_diffA_response','slider_diffB_response','slider_diffC_response'], axis = 1).reset_index(drop=True)  
    
    #get looking time for 1,2,3 choosen category
    trials.loc[trials['decision1_category']=='A','f_category1'] = trials['f_categoryA']
    trials.loc[trials['decision1_category']=='B','f_category1'] = trials['f_categoryB']   
    trials.loc[trials['decision1_category']=='C','f_category1'] = trials['f_categoryC'] 
    
    trials.loc[trials['decision2_category']=='A','f_category2'] = trials['f_categoryA']
    trials.loc[trials['decision2_category']=='B','f_category2'] = trials['f_categoryB']   
    trials.loc[trials['decision2_category']=='C','f_category2'] = trials['f_categoryC'] 
    
    trials.loc[trials['decision3_category']=='A','f_category3'] = trials['f_categoryA']
    trials.loc[trials['decision3_category']=='B','f_category3'] = trials['f_categoryB']   
    trials.loc[trials['decision3_category']=='C','f_category3'] = trials['f_categoryC'] 
    
#TRANSLATIONFOR GAZE DATA PLOTS     
    V1 = trials[['decision1_category','decision2_category','decision3_category','clicked_optionA','clicked_optionB','clicked_optionC','A1pos','A2pos','A3pos','B1pos','B2pos','B3pos','C1pos','C2pos','C3pos']] 
        
    V1.loc[trials['decision1_category']=='A','decision1_1category'] = V1['clicked_optionA']
    V1.loc[trials['decision1_category']=='B','decision1_1category'] = V1['clicked_optionB']
    V1.loc[trials['decision1_category']=='C','decision1_1category'] = V1['clicked_optionC']
    
    V1.loc[trials['decision2_category']=='A','decision2_1category'] = V1['clicked_optionA']
    V1.loc[trials['decision2_category']=='B','decision2_1category'] = V1['clicked_optionB']
    V1.loc[trials['decision2_category']=='C','decision2_1category'] = V1['clicked_optionC']
    
    V1.loc[trials['decision3_category']=='A','decision3_1category'] = V1['clicked_optionA']
    V1.loc[trials['decision3_category']=='B','decision3_1category'] = V1['clicked_optionB']
    V1.loc[trials['decision3_category']=='C','decision3_1category'] = V1['clicked_optionC']
    
    V1 = V1.drop(labels = ['clicked_optionA','clicked_optionB','clicked_optionC','decision1_category','decision2_category','decision3_category'], axis = 1).reset_index(drop=True)
   
    V1.A1pos = V1.A1pos.replace({'3_':''}, regex=True)
    V1.A2pos = V1.A2pos.replace({'3_':''}, regex=True)
    V1.A3pos = V1.A3pos.replace({'3_':''}, regex=True)
    V1.B1pos = V1.B1pos.replace({'3_':''}, regex=True)
    V1.B2pos = V1.B2pos.replace({'3_':''}, regex=True)
    V1.B3pos = V1.B3pos.replace({'3_':''}, regex=True)
    V1.C1pos = V1.C1pos.replace({'3_':''}, regex=True)
    V1.C2pos = V1.C2pos.replace({'3_':''}, regex=True)
    V1.C3pos = V1.C3pos.replace({'3_':''}, regex=True)
    
    cols = ['decision1_1category', 'decision2_1category', 'decision3_1category']
    V1[cols] = V1[cols].applymap(lambda lst: eval(lst)[0])
    
    V1.loc[V1['decision1_1category']=='A1','decision1_2category'] = 'A2'
    V1.loc[V1['decision1_1category']=='A1','decision1_3category'] = 'A3'
    
    V1.loc[V1['decision1_1category']=='A2','decision1_2category'] = 'A1'
    V1.loc[V1['decision1_1category']=='A2','decision1_3category'] = 'A3'
    
    V1.loc[V1['decision1_1category']=='A3','decision1_2category'] = 'A1'
    V1.loc[V1['decision1_1category']=='A3','decision1_3category'] = 'A2'
    
    
    V1.loc[V1['decision1_1category']=='B1','decision1_2category'] = 'B2'
    V1.loc[V1['decision1_1category']=='B1','decision1_3category'] = 'B3'
    
    V1.loc[V1['decision1_1category']=='B2','decision1_2category'] = 'B1'
    V1.loc[V1['decision1_1category']=='B2','decision1_3category'] = 'B3'
    
    V1.loc[V1['decision1_1category']=='B3','decision1_2category'] = 'B1'
    V1.loc[V1['decision1_1category']=='B3','decision1_3category'] = 'B2'
    
    
    V1.loc[V1['decision1_1category']=='C1','decision1_2category'] = 'C2'
    V1.loc[V1['decision1_1category']=='C1','decision1_3category'] = 'C3'
    
    V1.loc[V1['decision1_1category']=='C2','decision1_2category'] = 'C1'
    V1.loc[V1['decision1_1category']=='C2','decision1_3category'] = 'C3'
    
    V1.loc[V1['decision1_1category']=='C3','decision1_2category'] = 'C1'
    V1.loc[V1['decision1_1category']=='C3','decision1_3category'] = 'C2'
    
    
    
    V1.loc[V1['decision2_1category']=='A1','decision2_2category'] = 'A2'
    V1.loc[V1['decision2_1category']=='A1','decision2_3category'] = 'A3'
    
    V1.loc[V1['decision2_1category']=='A2','decision2_2category'] = 'A1'
    V1.loc[V1['decision2_1category']=='A2','decision2_3category'] = 'A3'
    
    V1.loc[V1['decision2_1category']=='A3','decision2_2category'] = 'A1'
    V1.loc[V1['decision2_1category']=='A3','decision2_3category'] = 'A2'
    
    
    V1.loc[V1['decision2_1category']=='B1','decision2_2category'] = 'B2'
    V1.loc[V1['decision2_1category']=='B1','decision2_3category'] = 'B3'
    
    V1.loc[V1['decision2_1category']=='B2','decision2_2category'] = 'B1'
    V1.loc[V1['decision2_1category']=='B2','decision2_3category'] = 'B3'
    
    V1.loc[V1['decision2_1category']=='B3','decision2_2category'] = 'B1'
    V1.loc[V1['decision2_1category']=='B3','decision2_3category'] = 'B2'
    
    
    V1.loc[V1['decision2_1category']=='C1','decision2_2category'] = 'C2'
    V1.loc[V1['decision2_1category']=='C1','decision2_3category'] = 'C3'
    
    V1.loc[V1['decision2_1category']=='C2','decision2_2category'] = 'C1'
    V1.loc[V1['decision2_1category']=='C2','decision2_3category'] = 'C3'
    
    V1.loc[V1['decision2_1category']=='C3','decision2_2category'] = 'C1'
    V1.loc[V1['decision2_1category']=='C3','decision2_3category'] = 'C2'



    V1.loc[V1['decision3_1category']=='A1','decision3_2category'] = 'A2'
    V1.loc[V1['decision3_1category']=='A1','decision3_3category'] = 'A3'
    
    V1.loc[V1['decision3_1category']=='A2','decision3_2category'] = 'A1'
    V1.loc[V1['decision3_1category']=='A2','decision3_3category'] = 'A3'
    
    V1.loc[V1['decision3_1category']=='A3','decision3_2category'] = 'A1'
    V1.loc[V1['decision3_1category']=='A3','decision3_3category'] = 'A2'

    
    V1.loc[V1['decision3_1category']=='B1','decision3_2category'] = 'B2'
    V1.loc[V1['decision3_1category']=='B1','decision3_3category'] = 'B3'
    
    V1.loc[V1['decision3_1category']=='B2','decision3_2category'] = 'B1'
    V1.loc[V1['decision3_1category']=='B2','decision3_3category'] = 'B3'
    
    V1.loc[V1['decision3_1category']=='B3','decision3_2category'] = 'B1'
    V1.loc[V1['decision3_1category']=='B3','decision3_3category'] = 'B2'
    
    
    V1.loc[V1['decision3_1category']=='C1','decision3_2category'] = 'C2'
    V1.loc[V1['decision3_1category']=='C1','decision3_3category'] = 'C3'
    
    V1.loc[V1['decision3_1category']=='C2','decision3_2category'] = 'C1'
    V1.loc[V1['decision3_1category']=='C2','decision3_3category'] = 'C3'
    
    V1.loc[V1['decision3_1category']=='C3','decision3_2category'] = 'C1'
    V1.loc[V1['decision3_1category']=='C3','decision3_3category'] = 'C2'
    
    x = ['decision1_1category','decision1_2category','decision1_3category','decision2_1category','decision2_2category','decision2_3category','decision3_1category','decision3_2category','decision3_3category']
     
    for i in x:
        V1.loc[V1[i]=='A1',i] = V1['A1pos']
        V1.loc[V1[i]=='A2',i] = V1['A2pos']
        V1.loc[V1[i]=='A3',i] = V1['A3pos']
        
        V1.loc[V1[i]=='B1',i] = V1['B1pos']
        V1.loc[V1[i]=='B2',i] = V1['B2pos']
        V1.loc[V1[i]=='B3',i] = V1['B3pos']
        
        V1.loc[V1[i]=='C1',i] = V1['C1pos']
        V1.loc[V1[i]=='C2',i] = V1['C2pos']
        V1.loc[V1[i]=='C3',i] = V1['C3pos']
  
    V1 = V1.drop(labels = ['A1pos','A2pos','A3pos','B1pos','B2pos','B3pos','C1pos','C2pos','C3pos'], axis = 1).reset_index(drop=True)
        
    trials['decision1_1category'] = V1['decision1_1category']
    trials['decision1_2category'] = V1['decision1_2category']
    trials['decision1_3category'] = V1['decision1_3category']
    
    trials['decision2_1category'] = V1['decision2_1category']
    trials['decision2_2category'] = V1['decision2_2category']
    trials['decision2_3category'] = V1['decision2_3category']
    
    trials['decision3_1category'] = V1['decision3_1category']
    trials['decision3_2category'] = V1['decision3_2category']
    trials['decision3_3category'] = V1['decision3_3category']
    
    #Add mean values for participant to results frame
    results.loc[results['participant']==trials['participant'].mean(),'validgaze_percent'] = trials['valid_datapoints'].mean()
    
    results_2.loc[results['participant']==trials['participant'].mean(),'W'] = trials['W'].mean()
    results_2.loc[results['participant']==trials['participant'].mean(),'B'] = trials['B'].mean()
    
    results_2.loc[results['participant']==trials['participant'].mean(),'f_category1'] = trials['f_category1'].mean()
    results_2.loc[results['participant']==trials['participant'].mean(),'f_category2'] = trials['f_category2'].mean()
    results_2.loc[results['participant']==trials['participant'].mean(),'f_category3'] = trials['f_category3'].mean()

    
    results_2.loc[results['participant']==trials['participant'].mean(),'decisiontime1'] = trials['decisiontime1'].mean()
    results_2.loc[results['participant']==trials['participant'].mean(),'decisiontime2'] = trials['decisiontime2'].mean()
    results_2.loc[results['participant']==trials['participant'].mean(),'decisiontimetotal'] = trials['decisiontimetotal'].mean()
    
    results_2.loc[results['participant']==trials['participant'].mean(),'category1_imp'] = trials['category1_imp'].mean()
    results_2.loc[results['participant']==trials['participant'].mean(),'category2_imp'] = trials['category2_imp'].mean()
    results_2.loc[results['participant']==trials['participant'].mean(),'category3_imp'] = trials['category3_imp'].mean()
    
    results_2.loc[results['participant']==trials['participant'].mean(),'category1_diff'] = trials['category1_diff'].mean()
    results_2.loc[results['participant']==trials['participant'].mean(),'category2_diff'] = trials['category2_diff'].mean()
    results_2.loc[results['participant']==trials['participant'].mean(),'category3_diff'] = trials['category3_diff'].mean()
    
    results_2.loc[results['participant']==trials['participant'].mean(),'WC1'] = trials['WC1'].mean()
    results_2.loc[results['participant']==trials['participant'].mean(),'WC2'] = trials['WC2'].mean()
    results_2.loc[results['participant']==trials['participant'].mean(),'WC3'] = trials['WC3'].mean()
    
def gaze_plt(subj_id):
    
    for index, trial in all_subj.iterrows():
        # select eyetracking data for the trial:
        gazedata = select_eyedata(trial, index, etData)

        # Vector with the positions of images ordered based on choice order of the trial
        translation = [
            int(trial.decision1_1category), int(trial.decision1_2category),
            int(trial.decision1_3category), int(trial.decision2_1category),
            int(trial.decision2_2category), int(trial.decision2_3category),
            int(trial.decision3_1category), int(trial.decision3_2category),
            int(trial.decision3_3category)
        ]

        # The first selected option of the first category should be on top of the graph so it becomes number 9 and so on
        gazedata.loc[gazedata['roi'] == translation[0], 'graph'] = 9
        gazedata.loc[gazedata['roi'] == translation[1], 'graph'] = 8
        gazedata.loc[gazedata['roi'] == translation[2], 'graph'] = 7
        gazedata.loc[gazedata['roi'] == translation[3], 'graph'] = 6
        gazedata.loc[gazedata['roi'] == translation[4], 'graph'] = 5
        gazedata.loc[gazedata['roi'] == translation[5], 'graph'] = 4
        gazedata.loc[gazedata['roi'] == translation[6], 'graph'] = 3
        gazedata.loc[gazedata['roi'] == translation[7], 'graph'] = 2
        gazedata.loc[gazedata['roi'] == translation[8], 'graph'] = 1

        # DataFrame with time and looked at location
        plot_df = pd.DataFrame([gazedata.graph, gazedata.time]).transpose()
        plot_df = plot_df.dropna()
        plot_df = plot_df.reset_index(drop=True)
        plt.figure()  # gaze plt
        # change starting time of images to 0 (eg. images started at 334 seconds of the experiment so for a single plot this becomes our t 0 since it is the beginning of the trial)
        plt.plot(plot_df['time'] - trial['et_decision_start'], plot_df['graph'], '-', linewidth="0.5", color="grey")
        plt.plot(plot_df['time'] - trial['et_decision_start'], plot_df['graph'], '.', markersize=2, color="black")

        # add colours for categories
        plt.axhspan(6.5, 9.5, facecolor='lightblue', alpha=0.7, label='Options of Category 1')
        plt.axhspan(3.5, 6.5, facecolor='deepskyblue', alpha=0.8, label='Options of Category 2')
        plt.axhspan(0.5, 3.5, facecolor='dodgerblue', alpha=0.3, label='Options of Category 3')
        plt.axvspan(0, trial['decisiontimetotal'], facecolor='grey', alpha=0.2)
        #plt.axvspan(trial['decisiontime1'], trial['decisiontime2'], facecolor='grey', alpha=0.15,
                    #label='Decision Phases')
        plt.axvspan(trial['decisiontime2'], trial['decisiontimetotal'] + 1, facecolor='grey', alpha=0.3)
        plt.axvline(x=trial['decisiontime1'])
        plt.axvline(x=trial['decisiontime2'])
        plt.axvline(x=trial['decisiontimetotal'])

        # set markers at the time point of a decision with different colors
        plt.plot(trial['decisiontime1'], 9, marker="o", markersize=12, markeredgecolor="black",
                 markerfacecolor="red", label='1st decision')
        plt.plot(trial['decisiontime2'], 6, marker="o", markersize=12, markeredgecolor="black",
                 markerfacecolor="green", label='2nd decision')
        plt.plot(trial['decisiontimetotal'], 3, marker="o", markersize=12, markeredgecolor="black",
                 markerfacecolor="blue", label='3rd decision')

        plt.yticks([1, 2, 3, 4, 5, 6, 7, 8, 9],
                   ['C 3 (C)', 'C 3 (B)', 'C 3 selected (A)', 'C 2 (C)', 'C 2 (B)', 'C 2 selected (A)', 'C 1 (C)',
                    'C 1 (B)', 'C 1 selected (A)'])
        plt.ylim([0.5, 9.5])
        plt.legend(bbox_to_anchor=(1, 0.65))
        plt.ylabel('Options')
        plt.xlabel('Time')
        plt.xlim(0, (trial['et_decision_end'] + 1) - trial['et_decision_start'])
        #plt.title(f'Trial {trial["trial.thisN"]}')
        plt.grid(False)
    # Customize the appearance of the figure
    # Save the figure for each trial
        #fig.savefig(f'gazeplot_trial_{trial["trial.thisN"]}.png', dpi=300, bbox_inches='tight', transparent=True)
        #plt.close(fig)
        #plt.savefig('my_plot.png',format="png", dpi=900)
        #filename = f'plot_{index}.png'
        #plt.savefig(filename, format="png", dpi=300, bbox_inches='tight',transparent=True)
        #plt.close()

    # Save the figure
        #fig.savefig(, dpi=300, bbox_inches='tight', transparent=True)
        plt.show()

#participant=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
#participant=[29]
# Call the function for all participants  
              
#/Users/fatma/Desktop/Data Planning Experiment/Data/
for subj_id in participant:
    gaze_plt(subj_id)
# Call the function
gaze_plt(subj_id)


# use this code to get the data for one specific subject
# get_planning_data('/Users/leonies/Documents/Master Thesis/Data Pilot 3/','4') #input: path to data storage and number of participant
# prepare_csv()
# gaze_plt()
# /Users/fatma/Desktop/Data Planning Experiment/Data/
# use this code to get the data for all subjects and different time frames
for participant in participant:
    # use your own laptop path here where the data is saved
    get_planning_data('/Users/fatma/Desktop/Data 3/', participant)
    prepare_csv()
    all_subj = pd.concat([all_subj, trials])
    all_subj = all_subj.reset_index(drop=True)

    get_planning_data0('/Users/fatma/Desktop/Data 3/', participant)
    prepare_csv_0()
    all_subj_0 = pd.concat([all_subj_0, trials])
    all_subj_0 = all_subj_0.reset_index(drop=True)

    get_planning_data1('/Users/fatma/Desktop/Data 3/', participant)
    prepare_csv_1()
    all_subj_1 = pd.concat([all_subj_1, trials])
    all_subj_1 = all_subj_1.reset_index(drop=True)

    get_planning_data2('/Users/fatma/Desktop/Data 3/', participant)
    prepare_csv_2()
    all_subj_2 = pd.concat([all_subj_2, trials])
    all_subj_2 = all_subj_2.reset_index(drop=True)



# How is the decison tree built (difficulty and importance among categoires choice order)    
    
plot_size = (18, 4)
def decision_order_diff_imp():
    # Difficulty plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 4))  # Set the overall figure size
    
    data_diff = pd.melt(results[['category1_diff', 'category2_diff', 'category3_diff']])
    ax_diff = sns.boxplot(x="variable", y="value", data=data_diff, palette=["coral", "orangered", "lightsalmon"], ax=axes[0])  # Change ax=axes[1] to ax=axes[0]
    ax_diff.set_xticks([0, 1, 2])
    ax_diff.set_xticklabels(['C 1', 'C 2', 'C 3'], fontsize=18, fontname="Times New Roman")
    ax_diff.set_xlabel('Decision order', fontsize=18, fontname="Times New Roman")
    ax_diff.set_ylabel("Difficulty", fontsize=18, fontname="Times New Roman")
    ax_diff.set_ylim(0, 10)
    ax_diff.tick_params(axis='y', labelsize=18) 
    # Set font size for y-axis labels
    ax_diff.set_yticks([0, 2.5, 5, 7.5, 10])
    ax_diff.axvline(x=-0.5, color='k', linewidth=4)
    add_stat_annotation(ax_diff, data=data_diff, x="variable", y="value",
                        box_pairs=[('category1_diff', 'category2_diff'), ('category1_diff', 'category3_diff'), ('category3_diff', 'category2_diff')],
                        test='t-test_paired', text_format='star', loc='inside', verbose=2, fontsize=18)
    # Add small horizontal lines at each y-axis tick
    y_ticks = [0, 2.5, 5, 7.5, 10]
    ax_diff.set_yticks(y_ticks)
    for y in y_ticks:
        ax_diff.plot([-0.5, -0.45], [y, y], color='k', linewidth=2)
    ax_diff.grid(False)
    
    # Importance plot
    data_imp = pd.melt(results[['category1_imp', 'category2_imp', 'category3_imp']])
    ax_imp = sns.boxplot(x="variable", y="value", data=data_imp, palette=["coral", "orangered", "lightsalmon"], ax=axes[1])  # Change ax=axes[0] to ax=axes[1]
    ax_imp.set_xticks([0, 1, 2])
    ax_imp.set_xticklabels(['C 1', 'C 2', 'C 3'], fontsize=18, fontname="Times New Roman")
    ax_imp.set_xlabel('Decision order', fontsize=18, fontname="Times New Roman")
    ax_imp.set_ylabel("Importance", fontsize=18, fontname="Times New Roman")
    ax_imp.set_ylim(0, 10)
    ax_imp.tick_params(axis='y', labelsize=18)  # Set font size for y-axis labels
    ax_imp.set_yticks([0, 2.5, 5, 7.5, 10])
    ax_imp.axvline(x=-0.5, color='k', linewidth=4)
    add_stat_annotation(ax_imp, data=data_imp, x="variable", y="value", 
                        box_pairs=[('category1_imp', 'category2_imp'), ('category1_imp', 'category3_imp'), ('category3_imp', 'category2_imp')],
                        test='t-test_paired', text_format='star', loc='inside', verbose=2, fontsize=20)
    y_ticks = [0, 2.5, 5, 7.5, 10]
    ax_imp.set_yticks(y_ticks)
    # Add small horizontal lines at each y-axis tick
    for y in y_ticks:
        ax_imp.plot([-0.5, -0.45], [y, y], color='k', linewidth=2)
    ax_imp.grid(False)
    
    # Adjust layout and save the plots
    plt.tight_layout()
    fig.savefig("diff_imp_plots.png", format="png", dpi=900, transparent=True)
    plt.show()
# Add vertical line
decision_order_diff_imp()

# fraction of within category gaze switches vs between category (for each time frame seperatly)


def WB_each_subject():
    # Time frame: results_0 = stimulus onset until first choice
    # new df for each subject plot
    custom_palette_1 = sns.color_palette(['#6A5ACD', '#4B0082']) 
    # change wide to long format to use seaborn
    A = pd.melt(results_0[['W']])
    B = pd.melt(results_0[['participant']])
    C1 = pd.concat([A, B], axis=1)
    A = pd.melt(results_0[['B']])
    B = pd.melt(results_0[['participant']])
    C2 = pd.concat([A, B], axis=1)
    A = pd.concat([C1, C2], axis=0)

    A.columns.values[0] = "WB"
    A.columns.values[1] = "f_category"
    A.columns.values[2] = "name"
    A.columns.values[3] = "participant"

    # Create the first plot for 1st Decision
    fig, axes = plt.subplots(1,3,figsize=plot_size)  # Create a grid of subplots
    ax0 = sns.boxplot(x="WB", y="f_category", data=A, palette=custom_palette_1, ax=axes[0])
    
    ax0.set(xlabel='', ylabel='Fraction of gaze switches')
    ax0.set_title("1st Decision",fontname="Times New Roman",fontsize=18)
    ax0.set_xticks([0, 1])
    
    ax0.set_xticklabels(['Within category', 'Across category'], fontname="Times New Roman", fontsize=18)
    ax0.set_ylabel('Fraction of gaze switches', fontname="Times New Roman", fontsize=18)
    add_stat_annotation(ax0, x="WB", y="f_category", data=A,
                        box_pairs=[('W', 'B')],
                        test='t-test_paired', text_format='star', loc='inside', verbose=2, fontsize=18)
    # Add vertical line
    ax0.axvline(x=-0.5, color='k', linewidth=4)

    # Customize y-axis ticks and labels
    y_ticks = [0, 0.25, 0.5, 0.75 ,1]
    ax0.set_yticks(y_ticks)
    ax0.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'], fontsize=18)
    ax0.set_ylim(0, 1.25)
    ax0.grid(False)

    # Add small horizontal lines at each y-axis tick
    for y in y_ticks:
        ax0.plot([-0.5, -0.45], [y, y], color='k', linewidth=2)
    # Time frame: results_1 = first choice until second choice
    custom_palette_2 = sns.color_palette(['#708238', '#556B2F'])  # Olive and DarkOliveGreen

    A = pd.melt(results_1[['W']])
    B = pd.melt(results_1[['participant']])
    C1 = pd.concat([A, B], axis=1)
    A = pd.melt(results_1[['B']])
    B = pd.melt(results_1[['participant']])
    C2 = pd.concat([A, B], axis=1)
    A = pd.concat([C1, C2], axis=0)
    A.columns.values[0] = "WB"
    A.columns.values[1] = "f_category"
    A.columns.values[2] = "name"
    A.columns.values[3] = "participant"

    # Create the second plot for 2nd Decision
    ax1 = sns.boxplot(x="WB", y="f_category", data=A, palette=custom_palette_2, ax=axes[1])
    ax1.set(xlabel='', ylabel='')  # Clear labels
    ax1.set_title("2nd Decision",fontname="Times New Roman",fontsize=18)
    #ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Within category', 'Across category'],fontname="Times New Roman", fontsize=18)
    add_stat_annotation(ax1, x="WB", y="f_category", data=A,
                        box_pairs=[('W', 'B')],
                        test='t-test_paired', text_format='star', loc='inside', verbose=2, fontsize=18)
    #ax1.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    #ax1.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'], fontsize=18)  # Provide labels
    #ax1.set_ylim(0.05, 0.95)
    #ax1.grid(False)
    ax1.set_yticks([])  # Remove y-axis ticks
    ax1.set_ylim(0, 1.25)
    ax1.grid(False)
    cyan = '#00FFFF'

# Lighter shade of cyan
    light_cyan = '#E0FFFF'

    # Time frame: results_2 = second choice until first choice
    custom_palette_3 = sns.color_palette([light_cyan, cyan])  # Tiffany Blue and LightSkyBlue

    A = pd.melt(results_2[['W']])
    B = pd.melt(results_2[['participant']])
    C1 = pd.concat([A, B], axis=1)
    A = pd.melt(results_2[['B']])
    B = pd.melt(results_2[['participant']])
    C2 = pd.concat([A, B], axis=1)
    A = pd.concat([C1, C2], axis=0)
    A.columns.values[0] = "WB"
    A.columns.values[1] = "f_category"
    A.columns.values[2] = "name"
    A.columns.values[3] = "participant"

    # Create the third plot for 3rd Decision
    ax2 = sns.boxplot(x="WB", y="f_category", data=A, palette=custom_palette_3, ax=axes[2])
    ax2.set(xlabel='', ylabel='')  # Clear labels
    ax2.set_title("3rd Decision",fontname="Times New Roman",fontsize=18)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Within category', 'Across category'], fontname="Times New Roman", fontsize=18)
    add_stat_annotation(ax2, x="WB", y="f_category", data=A,
                        box_pairs=[('W', 'B')],
                        test='t-test_paired', text_format='star', loc='inside', verbose=2, fontsize=18)
    ax2.set_yticks([])  # Remove y-axis ticks
    ax2.set_ylim(0, 1.25)
    ax2.grid(False)
    

    plt.tight_layout()  # Adjust the layout
    plt.savefig("WB_all_decisions.png", format="png", dpi=900, transparent=True)  # Save the figure
    plt.show()

WB_each_subject()

# change dataframe format of looking times to long istead of wide   
def wideframe():
    global wideframe
  
    fl_rank1 = pd.melt(all_subj[['f_category1','participant']])
    half_df = len(fl_rank1) // 2
    fl_rankA = fl_rank1.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank1.iloc[(half_df):,].reset_index(drop=True)
    fl_rank1 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    
    fl_rank2 = pd.melt(all_subj[['f_category2','participant']])
    half_df = len(fl_rank2) // 2
    fl_rankA = fl_rank2.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank2.iloc[(half_df):,].reset_index(drop=True)
    fl_rank2 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    fl_rank2=fl_rank2.dropna(axis=1, how='all')
    

    fl_rank3 = pd.melt(all_subj[['f_category3','participant']])
    half_df = len(fl_rank3) // 2
    fl_rankA = fl_rank3.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank3.iloc[(half_df):,].reset_index(drop=True)
    fl_rank3 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    fl_rank3=fl_rank3.dropna(axis=1, how='all')
    
    frames = [fl_rank1,fl_rank2, fl_rank3]
    fl_rank = pd.concat(frames).reset_index(drop=True)
    fl_rank.columns.values[0] = "order"
    fl_rank.columns.values[1] = "f_category"
    fl_rank.columns.values[2] = "name"
    fl_rank.columns.values[3] = "participant"
    fl_rank.loc[fl_rank['order']=='f_category1','order'] = 1
    fl_rank.loc[fl_rank['order']=='f_category2','order'] = 2
    fl_rank.loc[fl_rank['order']=='f_category3','order'] = 3
    
      
    
    fl_rank1 = pd.melt(all_subj[['category1_diff','participant']])
    half_df = len(fl_rank1) // 2
    fl_rankA = fl_rank1.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank1.iloc[(half_df):,].reset_index(drop=True)
    fl_rank1 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    
    fl_rank2 = pd.melt(all_subj[['category2_diff','participant']])
    half_df = len(fl_rank2) // 2
    fl_rankA = fl_rank2.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank2.iloc[(half_df):,].reset_index(drop=True)
    fl_rank2 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    fl_rank2=fl_rank2.dropna(axis=1, how='all')

    fl_rank3 = pd.melt(all_subj[['category3_diff','participant']])
    half_df = len(fl_rank3) // 2
    fl_rankA = fl_rank3.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank3.iloc[(half_df):,].reset_index(drop=True)
    fl_rank3 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    fl_rank3=fl_rank3.dropna(axis=1, how='all')
    
    frames = [fl_rank1,fl_rank2, fl_rank3]
    diff = pd.concat(frames).reset_index(drop=True)
    diff.columns.values[0] = "order"
    diff.columns.values[1] = "diff_category"
    diff.columns.values[2] = "name"
    diff.columns.values[3] = "participant"
    diff.loc[fl_rank['order']=='category1_diff','order'] = 1
    diff.loc[fl_rank['order']=='category2_diff','order'] = 2
    diff.loc[fl_rank['order']=='category3_diff','order'] = 3
    diff = diff.drop(labels = ['order','participant'], axis = 1).reset_index(drop=True)
    
    fl_rank1 = pd.melt(all_subj[['category1_imp','participant']])
    half_df = len(fl_rank1) // 2
    fl_rankA = fl_rank1.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank1.iloc[(half_df):,].reset_index(drop=True)
    fl_rank1 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    
    fl_rank2 = pd.melt(all_subj[['category2_imp','participant']])
    half_df = len(fl_rank2) // 2
    fl_rankA = fl_rank2.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank2.iloc[(half_df):,].reset_index(drop=True)
    fl_rank2 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    #fl_rank2=fl_rank2.dropna(axis=1, how='all')


    fl_rank3 = pd.melt(all_subj[['category3_imp','participant']])
    half_df = len(fl_rank3) // 2
    fl_rankA = fl_rank3.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank3.iloc[(half_df):,].reset_index(drop=True)
    fl_rank3 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    #fl_rank3=fl_rank3.dropna(axis=1, how='all')
    
    frames = [fl_rank1,fl_rank2, fl_rank3]
    imp = pd.concat(frames).reset_index(drop=True)
    imp.columns.values[0] = "order"
    imp.columns.values[1] = "imp_category"
    imp.columns.values[2] = "name"
    imp.columns.values[3] = "participant"
    imp.loc[fl_rank['order']=='category1_imp','order1'] = 1
    imp.loc[fl_rank['order']=='category2_imp','order1'] = 2
    imp.loc[fl_rank['order']=='category3_imp','order1'] = 3
    imp = imp.drop(labels = ['order','participant'], axis = 1).reset_index(drop=True)
    
    #fl_rank1 = pd.melt(all_subj[['decision1_category','participant']])
    fl_rank1 = pd.melt(all_subj[['RT1','participant']])
    half_df = len(fl_rank1) // 2
    fl_rankA = fl_rank1.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank1.iloc[(half_df):,].reset_index(drop=True)
    fl_rank1 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    #fl_rank2 = pd.melt(all_subj[['decision2_category','participant']])
    fl_rank2 = pd.melt(all_subj[['RT2','participant']])
    half_df = len(fl_rank2) // 2
    fl_rankA = fl_rank2.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank2.iloc[(half_df):,].reset_index(drop=True)
    fl_rank2 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    #fl_rank2=fl_rank2.dropna(axis=1, how='all')


    #fl_rank3 = pd.melt(all_subj[['decision3_category','participant']])
    fl_rank3 = pd.melt(all_subj[['RT3','participant']])
    half_df = len(fl_rank3) // 2
    #df1 = df1.loc[~df1.index.duplicated(keep='first')]~df1.index.duplicated(keep='first')
    fl_rankA = fl_rank3.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank3.iloc[(half_df):,].reset_index(drop=True)
    fl_rank3 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    #fl_rank3=fl_rank3.dropna(axis=1, how='all')
    
    frames = [fl_rank1,fl_rank2, fl_rank3]
    RT =pd.concat(frames).reset_index(drop=True)
    RT.columns.values[0] = "order"
    #dec.columns.values[1] = "decision_category"
    RT.columns.values[1] = "RT"
    RT.columns.values[2] = "name"
    RT.columns.values[3] = "participant"
    RT.loc[fl_rank['order']=='RT1','order2'] = 1
    RT.loc[fl_rank['order']=='RT2','order2'] = 2
    RT.loc[fl_rank['order']=='RT3','order2'] = 3
    RT = RT.drop(labels = ['order','participant'], axis = 1).reset_index(drop=True)
    
    #fl_rank1 = pd.melt(all_subj[['gazeswitch within category','participant']])
    fl_rank1 = pd.melt(all_subj_0[['W','participant']])
    half_df = len(fl_rank1) // 2
    fl_rankA = fl_rank1.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank1.iloc[(half_df):,].reset_index(drop=True)
    fl_rank1 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    #fl_rank2 = pd.melt(all_subj[['decision2_category','participant']])
    fl_rank2 = pd.melt(all_subj_1[['W','participant']])
    half_df = len(fl_rank2) // 2
    fl_rankA = fl_rank2.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank2.iloc[(half_df):,].reset_index(drop=True)
    fl_rank2 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    #fl_rank2=fl_rank2.dropna(axis=1, how='all')


    #fl_rank3 = pd.melt(all_subj[['decision3_category','participant']])
    fl_rank3 = pd.melt(all_subj_2[['W','participant']])
    half_df = len(fl_rank3) // 2
    #df1 = df1.loc[~df1.index.duplicated(keep='first')]~df1.index.duplicated(keep='first')
    fl_rankA = fl_rank3.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank3.iloc[(half_df):,].reset_index(drop=True)
    fl_rank3 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    #fl_rank3=fl_rank3.dropna(axis=1, how='all')
    
    frames = [fl_rank1,fl_rank2, fl_rank3]
    WC=pd.concat(frames).reset_index(drop=True)
    WC.columns.values[0] = "order"
    #dec.columns.values[1] = "decision_category"
    WC.columns.values[1] = "WC"
    WC.columns.values[2] = "name"
    WC.columns.values[3] = "participant"
    WC.loc[fl_rank['order']=='W','order3'] = 1
    WC.loc[fl_rank['order']=='W','order3'] = 2
    WC.loc[fl_rank['order']=='W','order3'] = 3
    WC = WC.drop(labels = ['order','participant'], axis = 1).reset_index(drop=True)
    
    #within category the fraction of gaze switch
    
    
    #fl_rank1 = pd.melt(all_subj[['gazeswitchbetween','participant']])
    fl_rank1 = pd.melt(all_subj_0[['B','participant']])
    half_df = len(fl_rank1) // 2
    fl_rankA = fl_rank1.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank1.iloc[(half_df):,].reset_index(drop=True)
    fl_rank1 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    
    fl_rank2 = pd.melt(all_subj_1[['B','participant']])
    half_df = len(fl_rank2) // 2
    fl_rankA = fl_rank2.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank2.iloc[(half_df):,].reset_index(drop=True)
    fl_rank2 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    #fl_rank2=fl_rank2.dropna(axis=1, how='all')


    fl_rank3 = pd.melt(all_subj_2[['B','participant']])
    half_df = len(fl_rank3) // 2
    fl_rankA = fl_rank3.iloc[:(half_df),].reset_index(drop=True)
    fl_rankB = fl_rank3.iloc[(half_df):,].reset_index(drop=True)
    fl_rank3 = pd.concat([fl_rankA, fl_rankB], axis=1).reset_index(drop=True)
    #fl_rank3=fl_rank3.dropna(axis=1, how='all')
    
    frames = [fl_rank1,fl_rank2, fl_rank3]
    CB = pd.concat(frames).reset_index(drop=True)
    CB.columns.values[0] = "order"
    CB.columns.values[1] = "B"
    CB.columns.values[2] = "name"
    CB.columns.values[3] = "participant"
    CB.loc[fl_rank['order']=='B','order1'] = 1
    CB.loc[fl_rank['order']=='B','order1'] = 2
    CB.loc[fl_rank['order']=='B','order1'] = 3
    CB = CB.drop(labels = ['order','participant'], axis = 1).reset_index(drop=True)
    #CB between category the fraction of gaze switch
    
    

    
    wideframe = pd.concat([fl_rank, diff,imp,RT,WC,CB], axis=1).reset_index(drop=True)
    wideframe =  wideframe[['order','f_category','diff_category','imp_category','RT','WC','B','participant']].dropna(axis=1, how='all')
    wideframe['order'] = pd.to_numeric( wideframe['order'])
    wideframe['f_category'] = pd.to_numeric( wideframe['f_category'])
    #wideframe['decision_time'] = pd.categorical( wideframe['decision_time'])
    wideframe['RT'] = pd.to_numeric( wideframe['RT'])
    wideframe['WC'] = pd.to_numeric( wideframe['WC'])
    wideframe['CB'] = pd.to_numeric( wideframe['B'])
    wideframe['diff_category'] =pd.to_numeric( wideframe['diff_category'])
    wideframe['imp_category'] = pd.to_numeric( wideframe['imp_category'])
    wideframe['participant'] = pd.to_numeric( wideframe['participant'])

all_subj['RTTotal']=all_subj['RT1']+all_subj['RT2']+all_subj['RT3']
all_subj['imp_category']=all_subj['category1_imp']+all_subj['category2_imp']+all_subj['category3_imp']
all_subj['diff_category']=all_subj['category1_diff']+all_subj['category2_diff']+all_subj['category3_diff']
all_subj['imp_category_mean']=all_subj['imp_category']/3
all_subj['diff_category_mean']=all_subj['diff_category']/3
#wideframe.dropna(inplace=True)
#all_subj.dropna(inplace=True)
#FileThatIuseForTheMultinomialLogisticRegression
wideframe.to_csv('DataFrameMultinomialRegression.csv')
# looking times at categoires before any decision is made 
def time_looked_c123():
    global aov_0
    global aov_1
    global aov_2
    global c1_values_decision_2
    global  c1_values_decision_3
    global A
    fig, axes = plt.subplots(1, 3, figsize=plot_size, sharey=True)

    # First figure
    A = pd.melt(results_0[['f_category1']])
    B = pd.melt(results_0[['f_category2']])
    C = pd.melt(results_0[['f_category3']])
    A = pd.concat([A, B, C], axis=0)
    A.columns.values[0] = "123"
    A.columns.values[1] = "f_category"
    A['index'] = A.index
    print(A)

    aov_0 = pg.rm_anova(data=A, dv='f_category', within='123', subject='index', detailed=True)
    #print(tabulate(aov_1, headers='keys', tablefmt='grid'))

    lt_0 = sns.boxplot(x="123", y="f_category", data=A, palette=["#800080", "#9370DB", "#D8BFD8"], showfliers=False, ax=axes[0])
    lt_0.set_xlabel('Decision order', fontsize=18, fontname="Times New Roman")
    lt_0.set_ylabel('Fraction of looking time', fontsize=18, fontname="Times New Roman", ha='center')  # Center align the y-axis label
    lt_0.set_title("1st Decision", fontname="Times New Roman", fontsize=18)
    lt_0.set_xticks([0, 1, 2])
    lt_0.set_xticklabels(['C 1', 'C 2', 'C 3'], fontname="Times New Roman", fontsize=18)
    add_stat_annotation(lt_0, x="123", y="f_category", data=A, box_pairs=[('f_category1', 'f_category2'), ('f_category2', 'f_category3'), ('f_category1', 'f_category3')], test='t-test_paired', text_format='star', loc='inside', verbose=2, fontsize=18)
    y_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lt_0.set_yticks(y_ticks)
    #lt_0.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'], fontsize=18)

# Set y-axis limits

    lt_0.axvline(x=-0.5, color='k', linewidth=3)
    # Add small horizontal lines at each y-axis tick
    y_ticks = lt_0.get_yticks()
    for y in y_ticks:
        lt_0.plot([-0.5, -0.45], [y, y], color='k', linewidth=2)
    lt_0.set_ylim(-0.05, 1.5)

    lt_0.grid(False)

    # Second figure
    A = pd.melt(results_1[['f_category1']])
    B = pd.melt(results_1[['f_category2']])
    C = pd.melt(results_1[['f_category3']])
    A = pd.concat([A, B, C], axis=0)
    A.columns.values[0] = "123"
    A.columns.values[1] = "f_category"
    A['index'] = A.index

    aov_1 = pg.rm_anova(data=A, dv='f_category', within='123', subject='index', detailed=True)
 
    # Extracting values for c1 in the second decision
    c1_values_decision_2 = A[A['123'] == 'f_category1']['f_category']
    #print("Fraction of looking time for c1 in 2nd Decision:", c1_values_decision_2.values)

    lt_1 = sns.boxplot(x="123", y="f_category", data=A, palette=["#006400", "#228B22", "#98FB98"], showfliers=False, ax=axes[1])
    
    lt_1.set_xlabel('Decision order', fontsize=18, fontname="Times New Roman")
    lt_1.set_ylabel('', fontsize=18, fontname="Times New Roman", ha='center')  # Center align the y-axis label

    lt_1.set_title("2nd Decision", fontname="Times New Roman", fontsize=18, pad=10)  # Adjust the title position downward
    lt_1.set_xticks([0, 1, 2])
    lt_1.set_xticklabels(['C 1', 'C 2', 'C 3'], fontname="Times New Roman", fontsize=18)
    add_stat_annotation(lt_1, x="123", y="f_category", data=A, box_pairs=[('f_category1', 'f_category2'), ('f_category2', 'f_category3'), ('f_category1', 'f_category3')], test='t-test_paired', text_format='star', loc='inside', verbose=2, fontsize=18)
    lt_1.set_yticks([0.0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1])
    lt_1.set_yticklabels(['0','0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1'], fontsize=18)  # Provide labels
    lt_1.set_ylim(-0.05, 1.5)
    lt_1.grid(False)

    # Third figure
    A = pd.melt(results_2[['f_category1']])
    B = pd.melt(results_2[['f_category2']])
    C = pd.melt(results_2[['f_category3']])
    A = pd.concat([A, B, C], axis=0)
    A.columns.values[0] = "123"
    A.columns.values[1] = "f_category"
    A['index'] = A.index

    aov_2 = pg.rm_anova(data=A, dv='f_category', within='123', subject='index', detailed=True)
    # Extracting values for c1 in the third decision
    c1_values_decision_3 = A[A['123'] == 'f_category1']['f_category']
  

    lt_2 = sns.boxplot(x="123", y="f_category", data=A, palette=["#000080", "#4169E1", "#87CEEB"], showfliers=False, ax=axes[2])
    lt_2.set_xlabel('Decision order', fontsize=18, fontname="Times New Roman")
    lt_2.set_ylabel('', fontsize=18, fontname="Times New Roman", ha='center')  # Center align the y-axis label

    lt_2.set_title("3rd Decision", fontname="Times New Roman", fontsize=18, pad=10)  # Adjust the title position downward
    lt_2.set_xticks([0, 1, 2])
    lt_2.set_xticklabels(['C 1', 'C 2', 'C 3'], fontname="Times New Roman", fontsize=18)
    add_stat_annotation(lt_2, x="123", y="f_category", data=A, box_pairs=[('f_category1', 'f_category2'), ('f_category2', 'f_category3'), ('f_category1', 'f_category3')], test='t-test_paired', text_format='star', loc='inside', verbose=2, fontsize=18)
    lt_2.set_yticks([0.0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1])
    lt_2.set_yticklabels(['0','0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1'], fontsize=18)  # Provide labels
    lt_2.set_ylim(-0.05, 1.5)
    lt_2.grid(False)

    plt.tight_layout()
    plt.savefig("fractionoflookingtime.png", format="png", dpi=900, transparent=True)
    plt.show()

time_looked_c123()

def correlation_f_category_diff_category():
    participant=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    coef_diff = []
    ci_diff = []
    pval_diff=[]
    for participant in participant:
        bool_subject = all_subj['participant']== participant
        participant_df= all_subj[bool_subject]
    #print(participant_df)
 
        bool_subject = wideframe['participant']== participant
        participant_df= wideframe[bool_subject]
    #print(participant_df)
        c=pg.corr(participant_df.f_category,participant_df.diff_category, method="spearman")
        co=float(c['r'])
        pval=float(c['p-val'])
        coef_diff.append(co)
        interval = (((c['CI95%'][0][1])-(c['CI95%'][0][0]))/2)
        ci_diff.append((interval))
        pval_diff.append(pval)
    
    
    participant=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    plt.figure()
    ax=plt.figure()
    plt.xticks([1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    plt.bar(participant,coef_diff,yerr=ci_diff,capsize = 4,color=['sandybrown', 'grey', 'grey', 'sandybrown','sandybrown', 'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown','sandybrown','sandybrown','sandybrown','sandybrown','sandybrown','sandybrown','grey','sandybrown','sandybrown','sandybrown','sandybrown','sandybrown','sandybrown','sandybrown','grey','sandybrown','sandybrown','sandybrown','grey'],alpha = 0.8) 
    plt.ylabel('Rho coefficient',fontsize=16)
    plt.xlabel('Participants',fontsize=16)
    plt.title("correlation between fraction of looking and the difficulty of catgory")
    print(pval_diff)
def correlation_f_category_imp_category():
    participant=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

    coef_impor = []
    ci_impor = []
    pval_impor=[]
    for participant in participant:
        bool_subject = all_subj_0['participant']== participant
        participant_df= all_subj_0[bool_subject]
        bool_subject = wideframe['participant']== participant
        participant_df= wideframe[bool_subject]
        c=pg.corr(participant_df.f_category,participant_df.imp_category, method="spearman")
        co=float(c['r'])
        pval=(c['p-val'])
        coef_impor.append(co)
        interval = (((c['CI95%'][0][1])-(c['CI95%'][0][0]))/2)
        ci_impor.append((interval))
        pval_impor.append(pval)
        print(c)
    
  
    participant=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    plt.figure()
    ax=plt.figure()
    plt.xticks([1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
    plt.bar(participant,coef_impor,yerr=ci_impor,capsize = 4,color=['grey', 'grey', 'sandybrown', 'grey','grey', 'grey', 'grey', 'grey', 'grey','grey','sandybrown','sandybrown','sandybrown','grey','sandybrown','grey','sandybrown','grey','grey','grey','grey','sandybrown','grey','grey','grey','sandybrown','sandybrown','grey'],alpha=0.8) 
    plt.ylabel('Rho coefficient',fontsize=16)
    plt.xlabel('Participants',fontsize=16)
    plt.title("correlation between the fraction of looking and the importance of catagory")
    

def correlation_diff_category_imp_category():
    participant=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    p=[]
    coef_impor_diff = []
    ci_impor_diff = []
    pval_impor_diff=[]
    for participant in participant:
        bool_subject = all_subj['participant']== participant
        participant_df= all_subj[bool_subject]
        bool_subject = wideframe['participant']== participant
        participant_df= wideframe[bool_subject]
        c=pg.corr(participant_df.diff_category,participant_df.imp_category, method="spearman")
        co=float(c['r'])
        coef_impor_diff.append(co)
        pval=(c['p-val'])
        interval = (((c['CI95%'][0][1])-(c['CI95%'][0][0]))/2)
        ci_impor_diff.append((interval))
        p.append(participant_df)
        pval_impor_diff.append(pval)
        print(c)
        print(coef_impor_diff)
#pg.corr(participant_df.diff_category,participant_df.imp_category, method="spearman")    
    participant=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    plt.figure()
    ax=plt.figure()
    plt.xticks([1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    plt.bar(participant,coef_impor_diff,yerr=ci_impor_diff,capsize = 4,color=['grey', 'grey', 'sandybrown', 'grey','grey', 'grey', 'grey', 'grey', 'grey','grey','sandybrown','sandybrown','sandybrown','grey','sandybrown','grey','sandybrown','grey','grey','grey','grey','sandybrown','grey','grey','grey','sandybrown','sandybrown','grey'],alpha = 0.8) 
    plt.ylabel('Rho coefficient',fontsize=16)
    plt.xlabel('Participants',fontsize=16)
    plt.title("correlation between difficulty and importance of catagory ")
def correlation_RT_imp():
    participant=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    coef = []
    ci = []
    alpha=0.05
    ci_gaze_confidence = []
    
    for participant in participant:
        bool_subject = all_subj['participant']== participant
        participant_df= all_subj[bool_subject]
        bool_subject = wideframe['participant']== participant
        participant_df= wideframe[bool_subject]
        c=rho, pval= stats.spearmanr(participant_df.RT,participant_df.imp_category)
        #
        co=rho
        
        r_z = np.arctanh(rho)
        se = 1/np.sqrt(participant_df.RT.size-3)
        z = stats.norm.ppf(1-alpha/2)
        lo_z, hi_z = r_z-z*se, r_z+z*se
        #interval = (((c['CI95%'][0][1])-(c['CI95%'][0][0]))/2)
        lo, hi = (np.tanh((lo_z, hi_z))/2)
        ci_gaze=((hi-lo)/2)
        coef.append(co)
        pv = pval
        ci.append(pv)
        ci_gaze_confidence.append(ci_gaze)
        print(c)
       
#c=pg.corr(participant_df.slider_confidence_response,participant_df.W_B_total, method="spearman")
#c=rho, pval = stats.spearmanr(participant_df.slider_confidence_response,participant_df.W_B_total)
    
    participant=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    plt.figure()
    ax=plt.figure()
    plt.xticks([1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])

    plt.bar(participant,coef,capsize = 4,yerr=ci_gaze,color=['grey', 'grey', 'grey', 'grey','grey', 'grey', 'grey', 'grey', 'grey','grey','grey','grey','grey','grey','sandybrown','grey','sandybrown','sandybrown','sandybrown','grey','grey','grey','grey','grey','sandybrown','sandybrown','sandybrown','grey'],alpha = 0.8) 
    plt.ylabel('Rho coefficient',fontsize=16)
    plt.xlabel('Participants',fontsize=16)

    plt.title("correlation between the RT and imp_category by participant ")
def correlation_RT_diff():
    participant=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    p=[]
    coef_RT_diff = []
    ci_RT_diff = []
    pval_RT_diff=[]
    alpha=0.05
    for participant in participant:
        bool_subject = all_subj['participant']== participant
        participant_df= all_subj[bool_subject]
        bool_subject = wideframe['participant']== participant
        participant_df= wideframe[bool_subject]
        c=rho, pval= stats.spearmanr(participant_df.RT,participant_df.imp_category)
        #
        co=rho
        
        r_z = np.arctanh(rho)
        se = 1/np.sqrt(participant_df.RT.size-3)
        z = stats.norm.ppf(1-alpha/2)
        lo_z, hi_z = r_z-z*se, r_z+z*se
        #interval = (((c['CI95%'][0][1])-(c['CI95%'][0][0]))/2)
        lo, hi = (np.tanh((lo_z, hi_z))/2)
        ci_RT=((hi-lo)/2)
        coef_RT_diff.append(co)
        pv = pval
        #ci_RT_diff.append(pv)
        ci_RT_diff.append(ci_RT)
        print(c)
#c=pg.corr(participant_df.slider_confidence_response,participant_df.W_B_total, method="spearman")
#c=rho, pval = stats.spearmanr(participant_df.slider_confidence_response,participant_df.W_B_total)
    
    participant=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    plt.figure()
    ax=plt.figure()
    plt.xticks([1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    plt.bar(participant,coef_RT_diff,yerr=ci_RT_diff,capsize = 4,color=['grey', 'grey', 'grey', 'grey','grey', 'grey', 'grey', 'grey', 'grey','grey','grey','grey','grey','grey','grey','sandybrown','grey','sandybrown','grey','sandybrown','sandybrown','grey','grey','grey','grey','sandybrown','sandybrown','grey'],alpha = 0.8) 
    plt.ylabel('Rho coefficient',fontsize=16)
    plt.xlabel('Participants',fontsize=16)
    plt.title("correlation between RT and difficulty of catagory ")
def correlation_confidene_gazeswitch_category():
    participant=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    coef = []
    ci = []
    alpha=0.05
    ci_gaze_confidence = []
    
    for participant in participant:
        bool_subject = all_subj['participant']== participant
        participant_df= all_subj[bool_subject]
        #participant_df.dropna(inplace=True)
        c=rho, pval= stats.spearmanr(participant_df.slider_confidence_response,participant_df.W_B_total)
        #
        co=rho
        
        r_z = np.arctanh(rho)
        se = 1/np.sqrt(participant_df.slider_confidence_response.size-3)
        z = stats.norm.ppf(1-alpha/2)
        lo_z, hi_z = r_z-z*se, r_z+z*se
        #interval = (((c['CI95%'][0][1])-(c['CI95%'][0][0]))/2)
        lo, hi = (np.tanh((lo_z, hi_z))/2)
        ci_gaze=((hi-lo)/2)
        coef.append(co)
        pv = pval
        ci.append(pv)
        ci_gaze_confidence.append(ci_gaze)
        print(c)
       
#c=pg.corr(participant_df.slider_confidence_response,participant_df.W_B_total, method="spearman")
#c=rho, pval = stats.spearmanr(participant_df.slider_confidence_response,participant_df.W_B_total)
    
    participant=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    plt.figure()
    ax=plt.figure()
    plt.xticks([1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])

    plt.bar(participant,coef,capsize = 4,yerr=ci_gaze,color=['sandybrown', 'sandybrown', 'grey', 'sandybrown','sandybrown', 'sandybrown', 'grey', 'sandybrown', 'sandybrown','sandybrown','sandybrown','grey','sandybrown','sandybrown','grey','grey','sandybrown','sandybrown','grey','grey','grey','sandybrown','sandybrown','sandybrown','grey','sandybrown','sandybrown','sandybrown'],alpha = 0.8) 
    plt.ylabel('Rho coefficient',fontsize=16)
    plt.xlabel('Participants',fontsize=16)

    plt.title("correlation between the gazeswitch and confidence response by participant ")
# plt.grid(axis='y')

def probability_difficulity():
    bins = np.linspace(-10,10,10) 
    labels = np.round(np.linspace(-10,10,9) ,2) 
    wideframe['binned1'] = pd.cut(wideframe['diff_category'], bins=bins,labels = labels)
    valuesRL = np.unique(wideframe['binned1'])
    ratiosR_c1 = []

    sems_r = []    
    for valueRL in valuesRL:
        if sum(wideframe['binned1'] == valueRL) == 0:
           ratioR_c1 = 0.0000000000001

           ratiosR_c1.append(ratioR_c1)
        else:
          #sem_r = np.nanstd(wideframe[wideframe['binned'] == valueRL].response,axis=0)/np.sqrt(len(wideframe[wideframe['binned'] == valueRL].response))
           ratioR_c1 = sum((wideframe['binned1'] == valueRL) & (wideframe['order']==1))/sum((wideframe['binned1'] == valueRL))

           ratiosR_c1.append(ratioR_c1)
          #sems_r.append(sem_r)
    print(ratiosR_c1)
    new_list = [item for item in valuesRL if not(math.isnan(item)) == True]
    x = new_list
    y1 = ratiosR_c1

    ratiosR_c2 = []

    sems_r = [] 
    for valueRL in valuesRL:
        if sum(wideframe['binned1'] == valueRL) == 0:
           ratioR_c_r = 0.0000000000001

           ratiosR_c2.append(ratioR_c2)
        else:
          #sem_r = np.nanstd(wideframe[wideframe['binned'] == valueRL].response,axis=0)/np.sqrt(len(wideframe[wideframe['binned'] == valueRL].response))
           ratioR_c2 = sum((wideframe['binned1'] == valueRL) & (wideframe['order']==2))/sum((wideframe['binned1'] == valueRL))

           ratiosR_c2.append(ratioR_c2)
          #sems_r.append(sem_r)
    print(ratiosR_c2)
    y2=ratiosR_c2
    ratiosR_c3 = []

    sems_r = [] 
    for valueRL in valuesRL:
        if sum(wideframe['binned1'] == valueRL) == 0:
           ratioR_c3 = 0.0000000000001

           ratiosR_c3.append(ratioR_c3)
        else:
          #sem_r = np.nanstd(wideframe[wideframe['binned'] == valueRL].response,axis=0)/np.sqrt(len(wideframe[wideframe['binned'] == valueRL].response))
           ratioR_c3 = sum((wideframe['binned1'] == valueRL) & (wideframe['order']==3))/sum((wideframe['binned1'] == valueRL))
           ratiosR_c3.append(ratioR_c3)
          #sems_r.append(sem_r)
    print(ratiosR_c3)
    y3=ratiosR_c3
    plt.plot(new_list,ratiosR_c1,'ro-',label='cat1')
    plt.plot(new_list,ratiosR_c2,'go-',label='cat2')
    plt.plot(new_list,ratiosR_c3,'bo-',label='cat3')
    plt.xlabel('diif_category',fontsize = 15)
    plt.ylabel('probabilty',fontsize = 15)


def gaze_plt_t():
        for index, trial in all_subj.iterrows():
            # select eyetracking data for the trial:      
            gazedata = select_eyedata(trial,index,etData)
            
            #Vector with the positions of images ordered based on choice order of the trial
            translation = [int(trial.decision1_1category),int(trial.decision1_2category),int(trial.decision1_3category),int(trial.decision2_1category),int(trial.decision2_2category),int(trial.decision2_3category),int(trial.decision3_1category),int(trial.decision3_2category),int(trial.decision3_3category) ]

            # The first selected option of the first category shoudl be on top of the graph so it becomes number 9 and so on
            gazedata.loc[gazedata['roi']== translation[0],'graph'] = 9
            gazedata.loc[gazedata['roi']== translation[1],'graph'] = 8
            gazedata.loc[gazedata['roi']== translation[2],'graph'] = 7
            gazedata.loc[gazedata['roi']== translation[3],'graph'] = 6
            gazedata.loc[gazedata['roi']== translation[4],'graph'] = 5
            gazedata.loc[gazedata['roi']== translation[5],'graph'] = 4
            gazedata.loc[gazedata['roi']== translation[6],'graph'] = 3
            gazedata.loc[gazedata['roi']== translation[7],'graph'] = 2
            gazedata.loc[gazedata['roi']== translation[8],'graph'] = 1
            
            #dataframe with time and looked at location
            plot_df = pd.DataFrame([gazedata.graph, gazedata.time]).transpose()
            plot_df = plot_df.dropna()
            plot_df = plot_df.reset_index(drop=True)
            plt.figure() 
            plt.plot(plot_df.time,plot_df.graph)
            plt.plot(plot_df['time']-trial['et_decision_start'],plot_df['graph'], '-', linewidth="0.5", color="grey")
            plt.plot(plot_df['time']-trial['et_decision_start'],plot_df['graph'], '.', markersize=2, color="black")
            plt.axvline(x=trial['decisiontime1'], color='darkred')
            plt.axvline(x=trial['decisiontime2'], color='darkred')
            plt.axvline(x=trial['decisiontimetotal'],color='darkred')
            
            # set markers at the time point of a decision
            plt.plot(trial['decisiontime1'], 9, marker="o", markersize=12, markeredgecolor="black", markerfacecolor="white",label= 'option selection')
            plt.plot(trial['decisiontime2'], 6, marker="o", markersize=12, markeredgecolor="black", markerfacecolor="white")
            plt.plot(trial['decisiontimetotal'], 3, marker="o", markersize=12, markeredgecolor="black", markerfacecolor="white")
            
            plt.yticks([1, 2,3,4,5,6,7,8,9], ['C 3', 'C 3','C 3 selected','C 2', 'C 2','C 2 selected','C 1', 'C 1','C 1 selected'])
            plt.ylim([0.5, 9.5])
            plt.legend(bbox_to_anchor=(1, 0.65))
            plt.ylabel('Options') 
            plt.xlabel('Time')
            plt.xlim(0, trial['et_decision_end']+1 -trial['et_decision_start'])
            plt.show() 

for index, trial in all_subj.iterrows():
    # select eyetracking data for the trial:      
    gazedata = select_eyedata(trial,index,etData)
    
    #Vector with the positions of images ordered based on choice order of the trial
    translation = [int(trial.decision1_1category),int(trial.decision1_2category),int(trial.decision1_3category),int(trial.decision2_1category),int(trial.decision2_2category),int(trial.decision2_3category),int(trial.decision3_1category),int(trial.decision3_2category),int(trial.decision3_3category) ]

    # The first selected option of the first category shoudl be on top of the graph so it becomes number 9 and so on
    gazedata.loc[gazedata['roi']== translation[0],'graph'] = 9
    gazedata.loc[gazedata['roi']== translation[1],'graph'] = 8
    gazedata.loc[gazedata['roi']== translation[2],'graph'] = 7
    gazedata.loc[gazedata['roi']== translation[3],'graph'] = 6
    gazedata.loc[gazedata['roi']== translation[4],'graph'] = 5
    gazedata.loc[gazedata['roi']== translation[5],'graph'] = 4
    gazedata.loc[gazedata['roi']== translation[6],'graph'] = 3
    gazedata.loc[gazedata['roi']== translation[7],'graph'] = 2
    gazedata.loc[gazedata['roi']== translation[8],'graph'] = 1
    
    #dataframe with time and looked at location
    plot_df = pd.DataFrame([gazedata.graph, gazedata.time]).transpose()
    plot_df = plot_df.dropna()
    plot_df = plot_df.reset_index(drop=True)
    plt.figure() 
    plt.plot(plot_df.time,plot_df.graph)
    plt.plot(plot_df['time']-trial['et_decision_start'],plot_df['graph'], '-', linewidth="0.5", color="grey")
    plt.plot(plot_df['time']-trial['et_decision_start'],plot_df['graph'], '.', markersize=2, color="black")
    plt.axvline(x=trial['decisiontime1'], color='darkred')
    plt.axvline(x=trial['decisiontime2'], color='darkred')
    plt.axvline(x=trial['decisiontimetotal'],color='darkred')
    
    # set markers at the time point of a decision
    plt.plot(trial['decisiontime1'], 9, marker="o", markersize=12, markeredgecolor="black", markerfacecolor="white",label= 'option selection')
    plt.plot(trial['decisiontime2'], 6, marker="o", markersize=12, markeredgecolor="black", markerfacecolor="white")
    plt.plot(trial['decisiontimetotal'], 3, marker="o", markersize=12, markeredgecolor="black", markerfacecolor="white")
    
    plt.yticks([1, 2,3,4,5,6,7,8,9], ['C 3', 'C 3','C 3 selected','C 2', 'C 2','C 2 selected','C 1', 'C 1','C 1 selected'])
    plt.ylim([0.5, 9.5])
    plt.legend(bbox_to_anchor=(1, 0.65))
    plt.ylabel('Options') 
    plt.xlabel('Time')
    plt.xlim(0, trial['et_decision_end']+1 -trial['et_decision_start'])


def gazeplot_all():
    global df_filtered_0
    global appended_data_gaze_all

    appended_data_gaze_all = []

    for index, trial in all_subj.iterrows():
        # Select eyetracking data for the trial:
        gazedata = select_eyedata(trial, index, etData)

        # Vector with the positions of images ordered based on choice order of the trial
        translation = [int(trial.decision1_1category), int(trial.decision1_2category), int(trial.decision1_3category),
                       int(trial.decision2_1category), int(trial.decision2_2category), int(trial.decision2_3category),
                       int(trial.decision3_1category), int(trial.decision3_2category), int(trial.decision3_3category)]

        # The first selected option of the first category should be on top of the graph so it becomes number 9 and so on
        gazedata['graph'] = gazedata['roi'].map(dict(zip(translation, reversed(range(1, 10)))))

        # Dataframe with time and looked-at location
        plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                  'lr_x': gazedata.lr_x,
                                  'time': gazedata.time,
                                  'roi': gazedata.roi,
                                  'graph': gazedata.graph})

        plot_df_1 = plot_df_1.dropna().reset_index(drop=True)
        appended_data_gaze_all.append(plot_df_1)

        # Create a scatter plot for gaze x and gaze y
        plt.figure(figsize=(10, 8))
        #sns.kdeplot(data=appended_data_gaze_all[index], x='lr_x', y='lr_y', cmap='viridis', fill=True, thresh=0.05)
        plt.scatter(x=appended_data_gaze_all[index]['lr_x'], y=appended_data_gaze_all[index]['lr_y'],
                    c=appended_data_gaze_all[index]['graph'], cmap='viridis', s=50, marker='o', edgecolors='w', linewidth=0.5)
        
        plt.title('Gaze Heatmap with Choices')
        plt.xlabel('Gaze X')
        plt.ylabel('Gaze Y')
        plt.colorbar(label='Choice Number')

        plt.show()

def gazeplot_all_each_trial():
    global df_filtered_0
    global appended_data_gaze_all

    appended_data_gaze_all = []

    for index, trial in all_subj.iterrows():
        # Select eyetracking data for the trial:
        gazedata = select_eyedata(trial, index, etData)

        # Vector with the positions of images ordered based on choice order of the trial
        translation = [int(trial.decision1_1category), int(trial.decision1_2category), int(trial.decision1_3category),
                       int(trial.decision2_1category), int(trial.decision2_2category), int(trial.decision2_3category),
                       int(trial.decision3_1category), int(trial.decision3_2category), int(trial.decision3_3category)]

        # The first selected option of the first category should be on top of the graph so it becomes number 9 and so on
        gazedata['graph'] = gazedata['roi'].map(dict(zip(translation, reversed(range(1, 10)))))

        # Dataframe with time and looked-at location
        plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                  'lr_x': gazedata.lr_x,
                                  'time': gazedata.time,
                                  'roi': gazedata.roi,
                                  'graph': gazedata.graph})

        plot_df_1 = plot_df_1.dropna().reset_index(drop=True)
        appended_data_gaze_all.append(plot_df_1)

        # Create a scatter plot for gaze x and gaze y
        plt.figure(figsize=(10, 8))
        plt.scatter(x=appended_data_gaze_all[index]['lr_x'], y=appended_data_gaze_all[index]['lr_y'],
                    c=appended_data_gaze_all[index]['graph'], cmap='viridis', s=50, marker='o', edgecolors='w', linewidth=0.5)

        # Display arrow with the same number as the color map value for the first point of each position
        for _, row in appended_data_gaze_all[index].groupby('graph').head(1).iterrows():
            plt.annotate(str(row['graph']), (row['lr_x'], row['lr_y']),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, arrowprops=dict(arrowstyle='->'))

        plt.title('Gaze Heatmap with Choices')
        plt.xlabel('Gaze X')
        plt.ylabel('Gaze Y')
        plt.colorbar(label='Choice Number')

        plt.show()
gazeplot_all_each_trial()

#dataframe contain timenormalized trial number choice gaze 
appended_data_gaze_all = []

for index, trial in all_subj.iterrows():
    gazedata = select_eyedata(trial, index, etData)

    translation = [int(trial.decision1_1category), int(trial.decision1_2category), int(trial.decision1_3category),
                   int(trial.decision2_1category), int(trial.decision2_2category), int(trial.decision2_3category),
                   int(trial.decision3_1category), int(trial.decision3_2category), int(trial.decision3_3category)]

    gazedata['graph'] = gazedata['roi'].map(dict(zip(translation, reversed(range(1, 10)))))

    plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                              'lr_x': gazedata.lr_x,
                              'time': gazedata.time,
                              'roi': gazedata.roi,
                              'graph': gazedata.graph})
    
    # Check if 'trial.thisN' exists in trial data and contains valid values
    if 'trial.thisN' in trial and pd.notna(trial['trial.thisN']):
        plot_df_1['trial_number'] = trial['trial.thisN']
    else:
        print(f"Warning: 'trial.thisN' not found or contains invalid values for trial at index {index}")

    plot_df_1['time_norm'] = plot_df_1['time'] - trial['et_decision_start']

    plot_df_1 = plot_df_1.dropna().reset_index(drop=True)
    appended_data_gaze_all.append(plot_df_1)


# Concatenate all DataFrames within the list
combined_data = pd.concat(appended_data_gaze_all, ignore_index=True)

# Create a new column 'normalized_time' in combined_data
combined_data['normalized_time'] = 0.0  # Initialize with zeros

# Iterate over unique trial numbers
for trial_number in combined_data['trial_number'].unique():
    # Extract data for the current trial
    trial_data = combined_data[combined_data['trial_number'] == trial_number].copy()
    
    # Find the maximum time for the current trial
    max_time_trial = trial_data['time_norm'].max()
    
    # Normalize time within each trial and update the 'normalized_time' column
    combined_data.loc[combined_data['trial_number'] == trial_number, 'normalized_time'] = \
        (trial_data['time_norm'] - trial_data['time_norm'].iloc[0]) / max_time_trial

# Group by the 'graph' column and create a dictionary of DataFrames for each trial
trial_data_dict = {trial: group[['lr_x', 'lr_y', 'graph','normalized_time']] for trial, group in combined_data.groupby('trial_number')}


# Define the categories based on the choice values
categories = {
    9: 'Category 1',
    8: 'Category 1',
    7: 'Category 1',
    6: 'Category 2',
    5: 'Category 2',
    4: 'Category 2',
    3: 'Category 3',
    2: 'Category 3',
    1: 'Category 3',
}

# Create a new column 'category' based on the 'choice' column
combined_data['category'] = combined_data['graph'].map(categories)


# Initialize empty DataFrames for each category
category_1_data = pd.DataFrame(columns=['trial_number', 'lr_x', 'lr_y','normalized_time'])
category_2_data = pd.DataFrame(columns=['trial_number', 'lr_x', 'lr_y','normalized_time'])
category_3_data = pd.DataFrame(columns=['trial_number', 'lr_x', 'lr_y','normalized_time'])

# Group the data by 'trial_number' and 'category' and collect gaze data for each category within each trial
grouped_data = combined_data.groupby(['trial_number', 'category'])

for (trial_number, category), group in grouped_data:
    # Select the appropriate DataFrame based on the category
    if category == 'Category 1':
        category_1_data = pd.concat([category_1_data, group[['trial_number', 'lr_x', 'lr_y','normalized_time']]], ignore_index=True)
    elif category == 'Category 2':
        category_2_data = pd.concat([category_2_data, group[['trial_number', 'lr_x', 'lr_y','normalized_time']]], ignore_index=True)
    elif category == 'Category 3':
        category_3_data = pd.concat([category_3_data, group[['trial_number', 'lr_x', 'lr_y','normalized_time']]], ignore_index=True)
# Plot heatmaps for each category
categories_data = [category_1_data, category_2_data, category_3_data]
category_names = ['Category 1', 'Category 2', 'Category 3']
mean_results = pd.DataFrame()

# Iterate over each category DataFrame
for category_idx, category_data in enumerate(categories_data, start=1):
    # Group by 'trial_number' and calculate the mean for each trial
    category_means = category_data.groupby('trial_number').agg({
        'lr_x': 'mean',
        'lr_y': 'mean'
    }).reset_index()

    # Rename columns to include the category index
    category_means.columns = [f'category_{category_idx}_{col}' if col.startswith('lr') else col for col in category_means.columns]

    # Merge with the mean_results DataFrame
    if mean_results.empty:
        mean_results = category_means
    else:
        mean_results = mean_results.merge(category_means, on='trial_number')


# Assuming mean_results is defined with appropriate column names

# Create a figure with three subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 8))

# Plot KDE for category 1
sns.kdeplot(x=mean_results['category_1_lr_x'], y=mean_results['category_1_lr_y'], cmap='magma', fill=True, cbar=True, ax=axes[0])
axes[0].set_title('Category 1')

# Plot KDE for category 2
sns.kdeplot(x=mean_results['category_2_lr_x'], y=mean_results['category_2_lr_y'], cmap='magma', fill=True, cbar=True, ax=axes[1])
axes[1].set_title('Category 2')

# Plot KDE for category 3
sns.kdeplot(x=mean_results['category_3_lr_x'], y=mean_results['category_3_lr_y'], cmap='magma', fill=True, cbar=True, ax=axes[2])
axes[2].set_title('Category 3')

# Set common labels for all subplots
for ax in axes:
    ax.set_xlabel('Mean Gaze X')
    ax.set_ylabel('Mean Gaze Y')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()



# Plot the distribution of the fraction of trials for each number of looked-at items
def gazeplot_single_subject_distribution_fraction_trials_lookeditems(subj_id):
    # Lists to store the counts of looked-at items for each trial
    global looked_at_counts
    global appended_data_gaze_single
    global plot_df_1
    plot_df_1=[]
    looked_at_counts = []
    appended_data_gaze_single = []

    plt.figure(figsize=(10, 8))  # Create the figure outside the loop

    subj_data = all_subj[all_subj['participant'] == subj_id]

    for index, trial in subj_data.iterrows():
        # Select eyetracking data for the trial:
        gazedata = select_eyedata(trial, index, etData)

        # Vector with the appended_data_gaze_single of images ordered based on choice order of the trial
        translation = [int(trial.decision1_1category), int(trial.decision1_2category), int(trial.decision1_3category),
                       int(trial.decision2_1category), int(trial.decision2_2category), int(trial.decision2_3category),
                       int(trial.decision3_1category), int(trial.decision3_2category), int(trial.decision3_3category)]

        # The first selected option of the first category should be on top of the graph so it becomes number 9 and so on
        gazedata['graph'] = gazedata['roi'].map(dict(zip(translation, reversed(range(1, 9)))))
        # Create a new column 'category' based on the 'graph' column
        gazedata['category'] = gazedata['graph'].map({
            9: 'Category 1',
            8: 'Category 1',
            7: 'Category 1',
            6: 'Category 2',
            5: 'Category 2',
            4: 'Category 2',
            3: 'Category 3',
            2: 'Category 3',
            1: 'Category 3',
        })
        
        plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                  'lr_x': gazedata.lr_x,
                                  'time': gazedata.time,
                                  'roi': gazedata.roi,
                                  'graph': gazedata.graph,
                                  'category': gazedata.category})  # Include the 'category' column in plot_df_1
        
        # Check if 'trial.thisN' exists in trial data and contains valid values
        if 'trial.thisN' in trial and pd.notna(trial['trial.thisN']):
            plot_df_1['trial_number'] = trial['trial.thisN']
        else:
            print(f"Warning: 'trial.thisN' not found or contains invalid values for trial at index {index}")
        
        plot_df_1['time_norm'] = plot_df_1['time'] - trial['et_decision_start']
        
        # Add 'RT1', 'RT2', 'RT3' columns to plot_df_1
        for category_num in range(1, 4):
            category_choice_time_col = f'RT{category_num}'
            category_choice_time = all_subj.at[index, category_choice_time_col]
            plot_df_1[category_choice_time_col] = category_choice_time
        
        



        # Dataframe with time and looked-at location
        plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                  'lr_x': gazedata.lr_x,
                                  'time': gazedata.time,
                                  'roi': gazedata.roi,
                                  'graph': gazedata.graph,
                                  'subject_id': [subj_id] * len(gazedata)})

        plot_df_1 = plot_df_1.dropna().reset_index(drop=True)
        appended_data_gaze_single.append(plot_df_1)

        # Store the count of looked-at items for each trial
        looked_at_counts.append(plot_df_1['graph'].nunique())

    # Plot the distribution of the fraction of trials for each number of looked-at items
    sns.histplot(looked_at_counts,  stat='probability', discrete=True)
    
    plt.title(f'Distribution of Looked-At Items in Trials (Subject {subj_id})')
    plt.xlabel('Number of Looked-At Items')
    plt.ylabel('Probability')

    plt.show()

# Specify the subject ID you are interested in
subject_of_interest = 20

# Call the function for the specified subject
gazeplot_single_subject_distribution_fraction_trials_lookeditems(subject_of_interest)
# Plot the distribution of the fraction of trials for each number of looked-at items for all subjects
def gazeplot_all_subjects_distribution_fraction_trials_lookeditems():
    plt.figure(figsize=(12, 8))  # Create the figure outside the loop
    
    # Lists to store the counts of looked-at items for each trial for all subjects
    all_looked_at_counts = []

    for subj_id in all_subj['participant'].unique():
        looked_at_counts = []
        appended_data_gaze_single = []

        subj_data = all_subj[all_subj['participant'] == subj_id]

        for index, trial in subj_data.iterrows():
            # Select eyetracking data for the trial:
            gazedata = select_eyedata(trial, index, etData)

            # Vector with the positions of images ordered based on choice order of the trial
            translation = [int(trial.decision1_1category), int(trial.decision1_2category), int(trial.decision1_3category),
                           int(trial.decision2_1category), int(trial.decision2_2category), int(trial.decision2_3category),
                           int(trial.decision3_1category), int(trial.decision3_2category), int(trial.decision3_3category)]

            # The first selected option of the first category should be on top of the graph so it becomes number 9 and so on
            gazedata['graph'] = gazedata['roi'].map(dict(zip(translation, reversed(range(1, 9)))))
            # Create a new column 'category' based on the 'graph' column
            gazedata['category'] = gazedata['graph'].map({
                9: 'Category 1',
                8: 'Category 1',
                7: 'Category 1',
                6: 'Category 2',
                5: 'Category 2',
                4: 'Category 2',
                3: 'Category 3',
                2: 'Category 3',
                1: 'Category 3',
            })
            
            plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                      'lr_x': gazedata.lr_x,
                                      'time': gazedata.time,
                                      'roi': gazedata.roi,
                                      'graph': gazedata.graph,
                                      'category': gazedata.category})  # Include the 'category' column in plot_df_1
            
            # Check if 'trial.thisN' exists in trial data and contains valid values
            if 'trial.thisN' in trial and pd.notna(trial['trial.thisN']):
                plot_df_1['trial_number'] = trial['trial.thisN']
            else:
                print(f"Warning: 'trial.thisN' not found or contains invalid values for trial at index {index}")
            
            plot_df_1['time_norm'] = plot_df_1['time'] - trial['et_decision_start']
            
            # Add 'RT1', 'RT2', 'RT3' columns to plot_df_1
            for category_num in range(1, 4):
                category_choice_time_col = f'RT{category_num}'
                category_choice_time = all_subj.at[index, category_choice_time_col]
                plot_df_1[category_choice_time_col] = category_choice_time
            
            plot_df_1 = plot_df_1.dropna().reset_index(drop=True)



            # Dataframe with time and looked-at location
            plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                      'lr_x': gazedata.lr_x,
                                      'time': gazedata.time,
                                      'roi': gazedata.roi,
                                      'graph': gazedata.graph,
                                      'subject_id': [subj_id] * len(gazedata)})

            plot_df_1 = plot_df_1.dropna().reset_index(drop=True)
            appended_data_gaze_single.append(plot_df_1)

            # Store the count of looked-at items for each trial
            looked_at_counts.append(plot_df_1['graph'].nunique())
        
        all_looked_at_counts.extend(looked_at_counts)

    # Plot the distribution of the fraction of trials for each number of looked-at items for all subjects
    sns.histplot(all_looked_at_counts, stat='probability', discrete=True, binwidth=1)
    
    plt.title('Distribution of Looked-At Items in Trials (All Subjects)')
    plt.xlabel('Number of Looked-At Items')
    plt.ylabel('Probability')

    plt.show()

# Call the function for all subjects
gazeplot_all_subjects_distribution_fraction_trials_lookeditems()

def Plot_distribution_fraction_trials_lookeditems():
    global appended_data_gaze_all
    global looked_at_counts
    global plot_df_1
 
    looked_at_counts = []
    appended_data_gaze_all = []
    

    plt.figure(figsize=(10, 8))

    # Lists to store the counts of looked-at items for each trial
    looked_at_counts = []

    for index, trial in all_subj.iterrows():
        # Select eyetracking data for the trial:
        gazedata = select_eyedata(trial, index, etData)

        # Vector with the positions of images ordered based on choice order of the trial
        translation = [
            int(trial.decision1_1category), int(trial.decision1_2category), int(trial.decision1_3category),
            int(trial.decision2_1category), int(trial.decision2_2category), int(trial.decision2_3category),
            int(trial.decision3_1category), int(trial.decision3_2category), int(trial.decision3_3category)
        ]

        for i, val in enumerate(translation):
            gazedata.loc[gazedata['roi'] == val, 'graph'] = 9 - i

        # Dataframe with time and looked-at location
        plot_df_1 = pd.DataFrame({
            'lr_y': gazedata.lr_y,
            'lr_x': gazedata.lr_x,
            'time': gazedata.time,
            'roi': gazedata.roi,
            'graph': gazedata.graph
        })
        # Create a new column 'category' based on the 'graph' column
        gazedata['category'] = gazedata['graph'].map({
            9: 'Category 1',
            8: 'Category 1',
            7: 'Category 1',
            6: 'Category 2',
            5: 'Category 2',
            4: 'Category 2',
            3: 'Category 3',
            2: 'Category 3',
            1: 'Category 3',
        })
        
        plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                  'lr_x': gazedata.lr_x,
                                  'time': gazedata.time,
                                  'roi': gazedata.roi,
                                  'graph': gazedata.graph,
                                  'category': gazedata.category})  # Include the 'category' column in plot_df_1
        
        # Check if 'trial.thisN' exists in trial data and contains valid values
        if 'trial.thisN' in trial and pd.notna(trial['trial.thisN']):
            plot_df_1['trial_number'] = trial['trial.thisN']
        else:
            print(f"Warning: 'trial.thisN' not found or contains invalid values for trial at index {index}")
        
        plot_df_1['time_norm'] = plot_df_1['time'] - trial['et_decision_start']
        
        # Add 'RT1', 'RT2', 'RT3' columns to plot_df_1
        for category_num in range(1, 4):
            category_choice_time_col = f'RT{category_num}'
            category_choice_time = all_subj.at[index, category_choice_time_col]
            plot_df_1[category_choice_time_col] = category_choice_time
        
        plot_df_1 = plot_df_1.dropna().reset_index(drop=True)

        plot_df_1 = plot_df_1.dropna().reset_index(drop=True)
        appended_data_gaze_all.append(plot_df_1)

        # Store the count of looked-at items for each trial
        looked_at_counts.append(plot_df_1['graph'].nunique())

    # Plot the distribution of the fraction of trials for each number of looked-at items
    sns.histplot(looked_at_counts, stat='probability', discrete=True, bins=range(1, 11), kde=False)

    plt.title('Distribution of Looked-At Items in Trials')
    plt.xlabel('Number of Looked-At Items')
    plt.ylabel('Probability')
    plt.xticks(range(1, 10))  # Set x-axis ticks for each number of looked-at items
    plt.savefig('Plot_distribution_fraction_trials_lookeditems.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

# Assuming required libraries and functions are defined elsewhere in your code.
# Make sure to adjust the parameters and include necessary imports accordingly.

Plot_distribution_fraction_trials_lookeditems()
def Plot_distribution_fraction_trials_lookeditems():
    global appended_data_gaze_all
    global looked_at_counts
    global plot_df_1
    looked_at_counts = []
    appended_data_gaze_all = []

    plt.figure(figsize=(10, 8))  # Create the figure outside the loop

    # Lists to store the counts of looked-at items for each trial
    looked_at_counts = []

    for trial_number, trial in all_subj.iterrows():  # Use trial_number directly from all_subj
        # Select eyetracking data for the trial:
        gazedata = select_eyedata(trial, trial_number, etData)

        # Vector with the positions of images ordered based on choice order of the trial
        translation = [int(trial.decision1_1category), int(trial.decision1_2category), int(trial.decision1_3category),
                       int(trial.decision2_1category), int(trial.decision2_2category), int(trial.decision2_3category),
                       int(trial.decision3_1category), int(trial.decision3_2category), int(trial.decision3_3category)]

        for index, _ in all_subj.iterrows():  # Use index as a dummy variable since it's not used inside the loop
            # select eyetracking data for the trial:
            gazedata = select_eyedata(trial, trial_number, etData)

            # Vector with the positions of images ordered based on choice order of the trial
            translation = [int(trial.decision1_1category), int(trial.decision1_2category),
                           int(trial.decision1_3category), int(trial.decision2_1category),
                           int(trial.decision2_2category), int(trial.decision2_3category),
                           int(trial.decision3_1category), int(trial.decision3_2category),
                           int(trial.decision3_3category)]

            # The first selected option of the first category should be on top of the graph so it becomes number 9 and so on
            gazedata.loc[gazedata['roi'] == translation[0], 'graph'] = 9
            gazedata.loc[gazedata['roi'] == translation[1], 'graph'] = 8
            gazedata.loc[gazedata['roi'] == translation[2], 'graph'] = 7
            gazedata.loc[gazedata['roi'] == translation[3], 'graph'] = 6
            gazedata.loc[gazedata['roi'] == translation[4], 'graph'] = 5

            gazedata.loc[gazedata['roi'] == translation[5], 'graph'] = 4
            gazedata.loc[gazedata['roi'] == translation[6], 'graph'] = 3
            gazedata.loc[gazedata['roi'] == translation[7], 'graph'] = 2
            gazedata.loc[gazedata['roi'] == translation[8], 'graph'] = 1

            # Dataframe with time and looked-at location
            plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                      'lr_x': gazedata.lr_x,
                                      'time': gazedata.time,
                                      'roi': gazedata.roi,
                                      'graph': gazedata.graph,
                                      'trial_number': trial_number + 1})  # Use trial_number from the outer loop

            plot_df_1 = plot_df_1.dropna().reset_index(drop=True)
            appended_data_gaze_all.append(plot_df_1)

            # Store the count of looked-at items for each trial
            looked_at_counts.append(plot_df_1['graph'].nunique())

    # Plot the distribution of the fraction of trials for each number of looked-at items
    sns.histplot(looked_at_counts, stat='probability', discrete=True)

    plt.title('Distribution of Looked-At Items in Trials')
    plt.xlabel('Number of Looked-At Items')
    plt.ylabel('Fraction of Trials')
    plt.savefig('Plot_distribution_fraction_trials_lookeditems.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

Plot_distribution_fraction_trials_lookeditems()

def Plot_distribution_fraction_trials_lookeditems():
    global appended_data_gaze_all
    global looked_at_counts
    global plot_df_1
    looked_at_counts=[]
    appended_data_gaze_all = []

    plt.figure(figsize=(10, 8))  # Create the figure outside the loop

    # Lists to store the counts of looked-at items for each trial
    looked_at_counts = []

    for index, trial in all_subj.iterrows():
        # Select eyetracking data for the trial:
        gazedata = select_eyedata(trial, index, etData)

        # Vector with the positions of images ordered based on choice order of the trial
        translation = [int(trial.decision1_1category), int(trial.decision1_2category), int(trial.decision1_3category),
                       int(trial.decision2_1category), int(trial.decision2_2category), int(trial.decision2_3category),
                       int(trial.decision3_1category), int(trial.decision3_2category), int(trial.decision3_3category)]
    for index, trial in all_subj.iterrows():
            # select eyetracking data for the trial:      
        gazedata = select_eyedata(trial,index,etData)
            
            #Vector with the positions of images ordered based on choice order of the trial
        translation = [int(trial.decision1_1category),int(trial.decision1_2category),int(trial.decision1_3category),int(trial.decision2_1category),int(trial.decision2_2category),int(trial.decision2_3category),int(trial.decision3_1category),int(trial.decision3_2category),int(trial.decision3_3category) ]

            # The first selected option of the first category shoudl be on top of the graph so it becomes number 9 and so on
        gazedata.loc[gazedata['roi']== translation[0],'graph'] = 9
        gazedata.loc[gazedata['roi']== translation[1],'graph'] = 8
        gazedata.loc[gazedata['roi']== translation[2],'graph'] = 7
        gazedata.loc[gazedata['roi']== translation[3],'graph'] = 6
        gazedata.loc[gazedata['roi']== translation[4],'graph'] = 5
            
        gazedata.loc[gazedata['roi']== translation[5],'graph'] = 4
        gazedata.loc[gazedata['roi']== translation[6],'graph'] = 3
        gazedata.loc[gazedata['roi']== translation[7],'graph'] = 2
        gazedata.loc[gazedata['roi']== translation[8],'graph'] = 1
            
            
        # Dataframe with time and looked-at location
        plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                  'lr_x': gazedata.lr_x,
                                  'time': gazedata.time,
                                  'roi': gazedata.roi,
                                  'graph': gazedata.graph})

        plot_df_1 = plot_df_1.dropna().reset_index(drop=True)
        appended_data_gaze_all.append(plot_df_1)

        # Store the count of looked-at items for each trial
        looked_at_counts.append(plot_df_1['graph'].nunique())

    # Plot the distribution of the fraction of trials for each number of looked-at items
    sns.histplot(looked_at_counts,  stat='probability', discrete=True)
    
    plt.title('Distribution of Looked-At Items in Trials')
    plt.xlabel('Number of Looked-At Items')
    plt.ylabel('Fraction of Trials')
    plt.savefig('Plot_distribution_fraction_trials_lookeditems.png', dpi=300, bbox_inches='tight',transparent=True)
    plt.show()

Plot_distribution_fraction_trials_lookeditems()

def gazeplot_all_heatmap_all_trial():
    global appended_data_gaze_all

    appended_data_gaze_all = []

    plt.figure(figsize=(12, 10))  # Adjust the figure size

    # Lists to store the x, y, and graph values for all trials
    all_lr_x = []
    all_lr_y = []
    all_graph = []

    for index, trial in all_subj.iterrows():
        # Select eyetracking data for the trial:
        gazedata = select_eyedata(trial, index, etData)

        # Vector with the positions of images ordered based on choice order of the trial
        translation = [int(trial.decision1_1category), int(trial.decision1_2category), int(trial.decision1_3category),
                       int(trial.decision2_1category), int(trial.decision2_2category), int(trial.decision2_3category),
                       int(trial.decision3_1category), int(trial.decision3_2category), int(trial.decision3_3category)]

        # The first selected option of the first category should be on top of the graph so it becomes number 9 and so on
        gazedata['graph'] = gazedata['roi'].map(dict(zip(translation, reversed(range(1, 10)))))

        # Dataframe with time and looked-at location
        plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                  'lr_x': gazedata.lr_x,
                                  'time': gazedata.time,
                                  'roi': gazedata.roi,
                                  'graph': gazedata.graph})

        plot_df_1 = plot_df_1.dropna().reset_index(drop=True)
        appended_data_gaze_all.append(plot_df_1)

        # Append the x, y, and graph values for the current trial to the lists
        all_lr_x.extend(plot_df_1['lr_x'])
        all_lr_y.extend(plot_df_1['lr_y'])
        all_graph.extend(plot_df_1['graph'])
    

    # Create a heatmap using plt.hist2d()
    plt.hist2d(x=all_lr_x, y=all_lr_y, bins=(25, 25), cmap='viridis', cmin=1)  # Adjust bins and cmap
    plt.colorbar(label='Density')  # Add a color bar

    # Display arrow with the same number as the color map value for the first point of each position
    for i, graph_value in enumerate(sorted(set(all_graph))):
        plt.annotate(str(graph_value), (all_lr_x[i], all_lr_y[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, arrowprops=dict(arrowstyle='->'))

    plt.title('Gaze Heatmap Across 9 Locations')
    plt.xlabel('Gaze X')
    plt.ylabel('Gaze Y')

    plt.show()

gazeplot_all_heatmap_all_trial()
def categorize_options(options):
    """
    Categorize options into groups based on predefined criteria.
    Adjust the criteria based on your specific requirements.
    """
    category_1 = [9, 8, 7]
    category_2 = [6, 5, 4]
    category_3 = [3, 2, 1]

    category = "Unknown"

    if any(option in category_1 for option in options):
        category = "Category 1"
    elif any(option in category_2 for option in options):
        category = "Category 2"
    elif any(option in category_3 for option in options):
        category = "Category 3"

    return category

def gazeplot_allFractionTrialsLookingDifferentCategories():
    global appended_data_gaze_all
    
    appended_data_gaze_all = []
    
    plt.figure(figsize=(12, 8))  # Adjust the figure size
    
    # Lists to store the counts of looked-at categories for each trial
    looked_at_categories_counts = {'Category 1': [], 'Category 2': [], 'Category 3': []}
    
    for index, trial in all_subj.iterrows():
        # Select eyetracking data for the trial:
        gazedata = select_eyedata(trial, index, etData)
        
        # Get the options for the current trial
        options = [
            trial.decision1_1category, trial.decision1_2category, trial.decision1_3category,
            trial.decision2_1category, trial.decision2_2category, trial.decision2_3category,
            trial.decision3_1category, trial.decision3_2category, trial.decision3_3category
        ]
        
        # Categorize the options into groups
        category = categorize_options(options)
        
        # Check if 'roi' is a valid column in gazedata
        if 'roi' in gazedata.columns:
            # Derive 'graph' from 'roi' using a mapping
            translation = [int(trial.decision1_1category), int(trial.decision1_2category), int(trial.decision1_3category),
                           int(trial.decision2_1category), int(trial.decision2_2category), int(trial.decision2_3category),
                           int(trial.decision3_1category), int(trial.decision3_2category), int(trial.decision3_3category)]
            
            gazedata['graph'] = gazedata['roi'].map(dict(zip(translation, reversed(range(1, 10)))))
            
            # Dataframe with time and looked-at location
            plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                      'lr_x': gazedata.lr_x,
                                      'time': gazedata.time,
                                      'roi': gazedata.roi,
                                      'graph': gazedata.graph})
            
            plot_df_1 = plot_df_1.dropna().reset_index(drop=True)
            appended_data_gaze_all.append(plot_df_1)
            
            # Append the counts to the corresponding category list
            for cat in looked_at_categories_counts.keys():
                looked_at_categories_counts[cat].append(1 if cat == category else 0)
        else:
            print(f"Warning: 'roi' column not found in trial {index}, skipping.")
    
    # Create a bar plot for the fraction of trials looking at different categories
    sns.barplot(data=pd.DataFrame(looked_at_categories_counts), ci=None)
    
    plt.title('Fraction of Trials Looking at Different Categories')
    plt.xlabel('Looked-At Category')
    plt.ylabel('Fraction of Trials')
    
    plt.show()
gazeplot_allFractionTrialsLookingDifferentCategories()




def gazeplot_all_heatmap_all_subjects():
    global appended_data_gaze_all

    appended_data_gaze_all = []

    # Set the seaborn style to 'white' to remove the grid
    sns.set_style("white")

    plt.figure(figsize=(10, 8))  # Create the figure outside the loop

    for subj_id in range(1, 29):  # Assuming subjects are numbered from 1 to 28
        subj_data = all_subj[all_subj['participant'] == subj_id]

        for index, trial in subj_data.iterrows():
            # Select eyetracking data for the trial:x xx¿?¡
            gazedata = select_eyedata(trial, index, etData)

            # Vector with the positions of images ordered based on choice order of the trial
            translation = [int(trial.decision1_1category), int(trial.decision1_2category), int(trial.decision1_3category),
                           int(trial.decision2_1category), int(trial.decision2_2category), int(trial.decision2_3category),
                           int(trial.decision3_1category), int(trial.decision3_2category), int(trial.decision3_3category)]

            # The first selected option of the first category should be on top of the graph so it becomes number 9 and so on
            gazedata['graph'] = gazedata['roi'].map(dict(zip(translation, reversed(range(1, 10)))))

            # Dataframe with time and looked-at location
            plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                      'lr_x': gazedata.lr_x,
                                      'time': gazedata.time,
                                      'roi': gazedata.roi,
                                      'graph': gazedata.graph})

            plot_df_1 = plot_df_1.dropna().reset_index(drop=True)
            appended_data_gaze_all.append(plot_df_1)

    # Concatenate gaze data across all subjects
    all_subjects_gaze_data = pd.concat(appended_data_gaze_all, ignore_index=True)

    # Create a heatmap with 'YlOrRd' colormap
    #sns.kdeplot(x=all_subjects_gaze_data['lr_x'], y=all_subjects_gaze_data['lr_y'], cmap='magma', fill=True, cbar=True)
    plt.plot(all_subjects_gaze_data['lr_x'],all_subjects_gaze_data['lr_y'],'.')
    plt.title('Gaze Heatmap for All Subjects and Trials')
    plt.xlabel('Gaze X')
    plt.ylabel('Gaze Y')

    # Save the plot without grid
    plt.savefig('gaze_heatmap_all_subj.png',dpi=300,transparent=True)  # Save the figure as an image file

    plt.show()

# Call the function
gazeplot_all_heatmap_all_subjects()

def gazeplot_individual_heatmaps():
    global appended_data_gaze_all
    
    appended_data_gaze_all = []
    
    # Set the seaborn style to 'white' to remove the grid
    sns.set_style("white")
    
    for subj_id in participant:
        plt.figure(figsize=(10, 8))  # Create a new figure for each subject
        
        subj_data = all_subj[all_subj['participant'] == subj_id]
        
        for index, trial in subj_data.iterrows():
            # Select eyetracking data for the trial:
            gazedata = select_eyedata(trial, index, etData)
            
            # Vector with the positions of images ordered based on the choice order of the trial
            translation = [int(trial.decision1_1category), int(trial.decision1_2category), int(trial.decision1_3category),
                           int(trial.decision2_1category), int(trial.decision2_2category), int(trial.decision2_3category),
                           int(trial.decision3_1category), int(trial.decision3_2category), int(trial.decision3_3category)]
            
            # The first selected option of the first category should be on top of the graph so it becomes number 9 and so on
            gazedata['graph'] = gazedata['roi'].map(dict(zip(translation, reversed(range(1, 10)))))
            
            # Dataframe with time and looked-at location
            plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                      'lr_x': gazedata.lr_x,
                                      'time': gazedata.time,
                                      'roi': gazedata.roi,
                                      'graph': gazedata.graph})
            
            plot_df_1 = plot_df_1.dropna().reset_index(drop=True)
            appended_data_gaze_all.append(plot_df_1)
        
        # Concatenate gaze data for the current subject
        subj_gaze_data = pd.concat(appended_data_gaze_all, ignore_index=True)
        
        # Create a heatmap with 'YlOrRd' colormap for the current subject
        ax = plt.subplot(1, 2, 1)  # Assuming you want two subplots: individual heatmap and all subjects heatmap
        sns.kdeplot(x=subj_gaze_data['lr_x'], y=subj_gaze_data['lr_y'], cmap='magma', fill=True, cbar=True)
        plt.title(f'Gaze Heatmap - Subject {subj_id}')
        plt.xlabel('Gaze X')
        plt.ylabel('Gaze Y')
        
        # Remove the grid for the individual heatmap
        sns.despine()
        
        # Save the individual heatmap with the subject name
        plt.savefig(f'gaze_heatmap_subject_{subj_id}.png')
        
        # Clear the subplot for the next iteration
        plt.clf()
    
    # Concatenate gaze data across all subjects
    all_subjects_gaze_data = pd.concat(appended_data_gaze_all, ignore_index=True)
    
    # Create a heatmap with 'YlOrRd' colormap for all subjects
    ax = plt.subplot(1, 2, 2)
    #sns.kdeplot(x=all_subjects_gaze_data['lr_x'], y=all_subjects_gaze_data['lr_y'], cmap='magma', fill=True, cbar=True)
    plt.plot(all_subjects_gaze_data['lr_x'],all_subjects_gaze_data['lr_y'])
    plt.title('Gaze Heatmap - All Subjects')
    plt.xlabel('Gaze X')
    plt.ylabel('Gaze Y')
    
    # Remove the grid for the all subjects heatmap
    sns.despine()
    
    # Save the all subjects heatmap
    plt.savefig('gaze_heatmap_all_subjects.png')
    
    plt.show()

# Call the function
gazeplot_individual_heatmaps()

def correlation_RT1_RT2_new():
    participant_ids = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

    coef = []
    ci_gaze_confidence = []
    bar_colors = []

    for participant_id in participant_ids:
        bool_subject = all_subj['participant'] == participant_id
        participant_df = all_subj[bool_subject]

        print(f'Participant {participant_id}')
        rho, pval = stats.spearmanr(participant_df['RT1'], participant_df['RT2'])

        r_z = np.arctanh(rho)
        se = 1 / np.sqrt(participant_df['RT1'].size - 3)
        z = stats.norm.ppf(1 - 0.05 / 2)
        lo_z, hi_z = r_z - z * se, r_z + z * se

        lo, hi = (np.tanh((lo_z, hi_z)) / 2)
        ci_gaze = ((hi - lo) / 2)

        coef.append(rho)
        ci_gaze_confidence.append(ci_gaze)
        print(pval)

        # Assign color based on p-value
        if pval < 0.05:
            bar_colors.append('sandybrown')
        else:
            bar_colors.append('grey')

    plt.figure()
    plt.bar(participant_ids, coef, yerr=ci_gaze_confidence, capsize=5, color=bar_colors, alpha=0.8)
    plt.ylabel('Rho coefficient', fontsize=16)
    plt.xlabel('Participants', fontsize=16)
    plt.title("Correlation between RT1 and RT22")

    plt.xticks([0, 5, 10, 15, 20, 25, 30])
    plt.grid(False)
    # Save the plot with good quality
    plt.savefig('correlation_RT1_RT2_plot.png', dpi=300, bbox_inches='tight',transparent=True)

    plt.show()

# Call the function
correlation_RT1_RT2_new()



def correlation_RT2_RT3_new():
    participant_ids = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

    coef = []
    ci_gaze_confidence = []
    bar_colors = []

    for participant_id in participant_ids:
        bool_subject = all_subj['participant'] == participant_id
        participant_df = all_subj[bool_subject]

        print(f'Participant {participant_id}')
        rho, pval = stats.spearmanr(participant_df['RT2'], participant_df['RT3'])

        r_z = np.arctanh(rho)
        se = 1 / np.sqrt(participant_df['RT2'].size - 3)
        z = stats.norm.ppf(1 - 0.05 / 2)
        lo_z, hi_z = r_z - z * se, r_z + z * se

        lo, hi = (np.tanh((lo_z, hi_z)) / 2)
        ci_gaze = ((hi - lo) / 2)

        coef.append(rho)
        ci_gaze_confidence.append(ci_gaze)
        print(pval)

        # Assign color based on p-value
        if pval < 0.05:
            bar_colors.append('sandybrown')
        else:
            bar_colors.append('grey')

    plt.figure()
    plt.bar(participant_ids, coef, yerr=ci_gaze_confidence, capsize=5, color=bar_colors, alpha=0.8)
    plt.ylabel('Rho coefficient', fontsize=16)
    plt.xlabel('Participants', fontsize=16)
    plt.title("Correlation between RT2 and RT3")

    plt.xticks([0, 5, 10, 15, 20, 25, 30])
    plt.grid(False)
    # Save the plot with good quality
    plt.savefig('correlation_RT2_RT3_plot.png', dpi=300, bbox_inches='tight',transparent=True)

    plt.show()

# Call the function
correlation_RT2_RT3_new()

def correlation_RT1_RT3_new():
    participant_ids = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

    coef = []
    ci_gaze_confidence = []
    bar_colors = []

    for participant_id in participant_ids:
        bool_subject = all_subj['participant'] == participant_id
        participant_df = all_subj[bool_subject]

        print(f'Participant {participant_id}')
        rho, pval = stats.spearmanr(participant_df['RT1'], participant_df['RT3'])

        r_z = np.arctanh(rho)
        se = 1 / np.sqrt(participant_df['RT2'].size - 3)
        z = stats.norm.ppf(1 - 0.05 / 2)
        lo_z, hi_z = r_z - z * se, r_z + z * se

        lo, hi = (np.tanh((lo_z, hi_z)) / 2)
        ci_gaze = ((hi - lo) / 2)

        coef.append(rho)
        ci_gaze_confidence.append(ci_gaze)
        print(pval)

        # Assign color based on p-value
        if pval < 0.05:
            bar_colors.append('sandybrown')
        else:
            bar_colors.append('grey')

    plt.figure()
    plt.bar(participant_ids, coef, yerr=ci_gaze_confidence, capsize=5, color=bar_colors, alpha=0.8)
    plt.ylabel('Rho coefficient', fontsize=16)
    plt.xlabel('Participants', fontsize=16)
    plt.title("Correlation between RT2 and RT3")

    plt.xticks([0, 5, 10, 15, 20, 25, 30])
    plt.grid(False)
    # Save the plot with good quality
    plt.savefig('correlation_RT1_RT3_plot.png', dpi=300, bbox_inches='tight',transparent=True)

    plt.show()

# Call the function
correlation_RT1_RT3_new()




def heatmap_all_category_allparticipants(participant):
    global appended_data_gaze_all
    global categories_data
    global grouped_data
    global combined_data

    appended_data_gaze_all = []

    for subj_id in participant:
        subj_data = all_subj[all_subj['participant'] == subj_id]

        for index, trial in subj_data.iterrows():
            gazedata = select_eyedata(trial, index, etData)

            translation = [int(trial.decision1_1category), int(trial.decision1_2category), int(trial.decision1_3category),
                           int(trial.decision2_1category), int(trial.decision2_2category), int(trial.decision2_3category),
                           int(trial.decision3_1category), int(trial.decision3_2category), int(trial.decision3_3category)]

            gazedata['graph'] = gazedata['roi'].map(dict(zip(translation, reversed(range(1, 10)))))

            plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                      'lr_x': gazedata.lr_x,
                                      'time': gazedata.time,
                                      'roi': gazedata.roi,
                                      'graph': gazedata.graph})
            
            # Check if 'trial.thisN' exists in trial data and contains valid values
            if 'trial.thisN' in trial and pd.notna(trial['trial.thisN']):
                plot_df_1['trial_number'] = trial['trial.thisN']
            else:
                print(f"Warning: 'trial.thisN' not found or contains invalid values for trial at index {index}")

            plot_df_1['time_norm'] = plot_df_1['time'] - trial['et_decision_start']

            plot_df_1 = plot_df_1.dropna().reset_index(drop=True)
            appended_data_gaze_all.append(plot_df_1)

    # Concatenate all DataFrames within the list
    combined_data = pd.concat(appended_data_gaze_all, ignore_index=True)

    # Create a new column 'normalized_time' in combined_data
    combined_data['normalized_time'] = 0.0  # Initialize with zeros

    # Iterate over unique trial numbers
    for trial_number in combined_data['trial_number'].unique():
        # Extract data for the current trial
        trial_data = combined_data[combined_data['trial_number'] == trial_number].copy()

        # Find the maximum time for the current trial
        max_time_trial = trial_data['time_norm'].max()

        # Normalize time within each trial and update the 'normalized_time' column
        combined_data.loc[combined_data['trial_number'] == trial_number, 'normalized_time'] = \
            (trial_data['time_norm'] - trial_data['time_norm'].iloc[0]) / max_time_trial

    # Group by the 'graph' column and create a dictionary of DataFrames for each trial
    trial_data_dict = {trial: group[['lr_x', 'lr_y', 'graph', 'normalized_time']] for trial, group in
                      combined_data.groupby('trial_number')}

    # Define the categories based on the choice values
    categories = {
        9: 'Category 1',
        8: 'Category 1',
        7: 'Category 1',
        6: 'Category 2',
        5: 'Category 2',
        4: 'Category 2',
        3: 'Category 3',
        2: 'Category 3',
        1: 'Category 3',
    }

    # Create a new column 'category' based on the 'choice' column
    combined_data['category'] = combined_data['graph'].map(categories)

    # Initialize empty DataFrames for each category
    category_1_data = pd.DataFrame(columns=['trial_number', 'lr_x', 'lr_y', 'normalized_time'])
    category_2_data = pd.DataFrame(columns=['trial_number', 'lr_x', 'lr_y', 'normalized_time'])
    category_3_data = pd.DataFrame(columns=['trial_number', 'lr_x', 'lr_y', 'normalized_time'])

    # Group the data by 'trial_number' and 'category' and collect gaze data for each category within each trial
    grouped_data = combined_data.groupby(['trial_number', 'category'])

    for (trial_number, category), group in grouped_data:
        # Select the appropriate DataFrame based on the category
        if category == 'Category 1':
            category_1_data = pd.concat([category_1_data, group[['trial_number', 'lr_x', 'lr_y', 'normalized_time']]],
                                       ignore_index=True)
        elif category == 'Category 2':
            category_2_data = pd.concat([category_2_data, group[['trial_number', 'lr_x', 'lr_y', 'normalized_time']]],
                                       ignore_index=True)
        elif category == 'Category 3':
            category_3_data = pd.concat([category_3_data, group[['trial_number', 'lr_x', 'lr_y', 'normalized_time']]],
                                       ignore_index=True)

    # Plot heatmaps for each category
    categories_data = [category_1_data, category_2_data, category_3_data]
    category_names = ['Category 1', 'Category 2', 'Category 3']
    mean_results = pd.DataFrame()

    # Iterate over each category DataFrame
    for category_idx, category_data in enumerate(categories_data, start=1):
        # Group by 'trial_number' and calculate the mean for each trial
        category_means = category_data.groupby('trial_number').agg({
            'lr_x': 'mean',
            'lr_y': 'mean'
        }).reset_index()

        # Rename columns to include the category index
        category_means.columns = [f'category_{category_idx}_{col}' if col.startswith('lr') else col for col in
                                  category_means.columns]

        # Merge with the mean_results DataFrame
        if mean_results.empty:
            mean_results = category_means
        else:
            mean_results = mean_results.merge(category_means, on='trial_number')

    # Create a figure with three subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 8))

    # Plot KDE for category 1
    sns.kdeplot(x=mean_results['category_1_lr_x'], y=mean_results['category_1_lr_y'], cmap='magma', fill=True,
                cbar=True, ax=axes[0])
    axes[0].set_title('Category 1')

    # Plot KDE for category 2
    sns.kdeplot(x=mean_results['category_2_lr_x'], y=mean_results['category_2_lr_y'], cmap='magma', fill=True,
                cbar=True, ax=axes[1])
    axes[1].set_title('Category 2')

    # Plot KDE for category 3
    sns.kdeplot(x=mean_results['category_3_lr_x'], y=mean_results['category_3_lr_y'], cmap='magma', fill=True,
                cbar=True, ax=axes[2])
    axes[2].set_title('Category 3')

    # Set common labels for all subplots
    for ax in axes:
        ax.set_xlabel('Mean Gaze X')
        ax.set_ylabel('Mean Gaze Y')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.grid(False)
    plt.savefig('heatmap_all_category_allparticipants.png', dpi=300, bbox_inches='tight',transparent=True)


    # Show the plot
    plt.show()

# Example: Call the function for participants 1, 2, and 3
heatmap_all_category_allparticipants([1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])

#heatmap_categories
def heatmap_category123_individual_participant(participant_id):
    global appended_data_gaze_all

    appended_data_gaze_all = []

    subj_data = all_subj[all_subj['participant'] == participant_id]

    for index, trial in subj_data.iterrows():
        gazedata = select_eyedata(trial, index, etData)

        translation = [int(trial.decision1_1category), int(trial.decision1_2category), int(trial.decision1_3category),
                       int(trial.decision2_1category), int(trial.decision2_2category), int(trial.decision2_3category),
                       int(trial.decision3_1category), int(trial.decision3_2category), int(trial.decision3_3category)]

        gazedata['graph'] = gazedata['roi'].map(dict(zip(translation, reversed(range(1, 10)))))

        plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                  'lr_x': gazedata.lr_x,
                                  'time': gazedata.time,
                                  'roi': gazedata.roi,
                                  'graph': gazedata.graph})
        
        # Check if 'trial.thisN' exists in trial data and contains valid values
        if 'trial.thisN' in trial and pd.notna(trial['trial.thisN']):
            plot_df_1['trial_number'] = trial['trial.thisN']
        else:
            print(f"Warning: 'trial.thisN' not found or contains invalid values for trial at index {index}")

        plot_df_1['time_norm'] = plot_df_1['time'] - trial['et_decision_start']

        plot_df_1 = plot_df_1.dropna().reset_index(drop=True)
        appended_data_gaze_all.append(plot_df_1)

    # Concatenate all DataFrames within the list
    combined_data = pd.concat(appended_data_gaze_all, ignore_index=True)

    # Create a new column 'normalized_time' in combined_data
    combined_data['normalized_time'] = 0.0  # Initialize with zeros

    # Iterate over unique trial numbers
    for trial_number in combined_data['trial_number'].unique():
        # Extract data for the current trial
        trial_data = combined_data[combined_data['trial_number'] == trial_number].copy()

        # Find the maximum time for the current trial
        max_time_trial = trial_data['time_norm'].max()

        # Normalize time within each trial and update the 'normalized_time' column
        combined_data.loc[combined_data['trial_number'] == trial_number, 'normalized_time'] = \
            (trial_data['time_norm'] - trial_data['time_norm'].iloc[0]) / max_time_trial

    # Group by the 'graph' column and create a dictionary of DataFrames for each trial
    trial_data_dict = {trial: group[['lr_x', 'lr_y', 'graph', 'normalized_time']] for trial, group in
                      combined_data.groupby('trial_number')}

    # Define the categories based on the choice values
    categories = {
        9: 'Category 1',
        8: 'Category 1',
        7: 'Category 1',
        6: 'Category 2',
        5: 'Category 2',
        4: 'Category 2',
        3: 'Category 3',
        2: 'Category 3',
        1: 'Category 3',
    }

    # Create a new column 'category' based on the 'choice' column
    combined_data['category'] = combined_data['graph'].map(categories)

    # Initialize empty DataFrames for each category
    category_1_data = pd.DataFrame(columns=['trial_number', 'lr_x', 'lr_y', 'normalized_time'])
    category_2_data = pd.DataFrame(columns=['trial_number', 'lr_x', 'lr_y', 'normalized_time'])
    category_3_data = pd.DataFrame(columns=['trial_number', 'lr_x', 'lr_y', 'normalized_time'])

    # Group the data by 'trial_number' and 'category' and collect gaze data for each category within each trial
    grouped_data = combined_data.groupby(['trial_number', 'category'])

    for (trial_number, category), group in grouped_data:
        # Select the appropriate DataFrame based on the category
        if category == 'Category 1':
            category_1_data = pd.concat([category_1_data, group[['trial_number', 'lr_x', 'lr_y', 'normalized_time']]],
                                       ignore_index=True)
        elif category == 'Category 2':
            category_2_data = pd.concat([category_2_data, group[['trial_number', 'lr_x', 'lr_y', 'normalized_time']]],
                                       ignore_index=True)
        elif category == 'Category 3':
            category_3_data = pd.concat([category_3_data, group[['trial_number', 'lr_x', 'lr_y', 'normalized_time']]],
                                       ignore_index=True)

    # Plot heatmaps for each category
    categories_data = [category_1_data, category_2_data, category_3_data]
    category_names = ['Category 1', 'Category 2', 'Category 3']
    mean_results = pd.DataFrame()

    # Iterate over each category DataFrame
    for category_idx, category_data in enumerate(categories_data, start=1):
        # Group by 'trial_number' and calculate the mean for each trial
        category_means = category_data.groupby('trial_number').agg({
            'lr_x': 'mean',
            'lr_y': 'mean'
        }).reset_index()

        # Rename columns to include the category index
        category_means.columns = [f'category_{category_idx}_{col}' if col.startswith('lr') else col for col in
                                  category_means.columns]

        # Merge with the mean_results DataFrame
        if mean_results.empty:
            mean_results = category_means
        else:
            mean_results = mean_results.merge(category_means, on='trial_number')

    # Create a figure with three subplots
    # Create a figure with three subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 8))
    #fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 8))

# Plot KDE for category 1
    sns.kdeplot(x=mean_results['category_1_lr_x'], y=mean_results['category_1_lr_y'], cmap='magma', fill=True,
            cbar=True, ax=axes[0])
    axes[0].set_title(f'Participant {participant_id} - Category 1')

# Plot KDE for category 2
    sns.kdeplot(x=mean_results['category_2_lr_x'], y=mean_results['category_2_lr_y'], cmap='magma', fill=True,
            cbar=True, ax=axes[1])
    axes[1].set_title(f'Participant {participant_id} - Category 2')

# Plot KDE for category 3
    sns.kdeplot(x=mean_results['category_3_lr_x'], y=mean_results['category_3_lr_y'], cmap='magma', fill=True,
            cbar=True, ax=axes[2])
    axes[2].set_title(f'Participant {participant_id} - Category 3')

# Set common labels for all subplots
    for ax in axes:
        ax.set_xlabel('Mean Gaze X')
        ax.set_ylabel('Mean Gaze Y')

# Adjust layout to prevent overlapping
    plt.tight_layout()

# Show the plot
    plt.show()




# Call the function for a specific participant (e.g., participant 1)
heatmap_category123_individual_participant(10)

def revisit_category1_ategory_2_itsturn(subject_id):
    global appended_data_revisits
    
    appended_data_revisits = []
    
    subj_data = all_subj[all_subj['participant'] == subject_id]
    
    results_df = pd.DataFrame(columns=['Subject', 'Trial', 'Fraction_Revisits_Category1'])
    
    for index, trial in subj_data.iterrows():
        gazedata = select_eyedata(trial, index, etData)
        
        translation = [int(trial.decision1_1category), int(trial.decision1_2category), int(trial.decision1_3category),
                       int(trial.decision2_1category), int(trial.decision2_2category), int(trial.decision2_3category),
                       int(trial.decision3_1category), int(trial.decision3_2category), int(trial.decision3_3category)]
        
        gazedata['graph'] = gazedata['roi'].map(dict(zip(translation, reversed(range(1, 10)))))
        
        # Create a new column 'category' based on the 'graph' column
        gazedata['category'] = gazedata['graph'].map({
            9: 'Category 1',
            8: 'Category 1',
            7: 'Category 1',
            6: 'Category 2',
            5: 'Category 2',
            4: 'Category 2',
            3: 'Category 3',
            2: 'Category 3',
            1: 'Category 3',
        })
        
        plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                  'lr_x': gazedata.lr_x,
                                  'time': gazedata.time,
                                  'roi': gazedata.roi,
                                  'graph': gazedata.graph,
                                  'category': gazedata.category})  # Include the 'category' column in plot_df_1
        
        # Check if 'trial.thisN' exists in trial data and contains valid values
        if 'trial.thisN' in trial and pd.notna(trial['trial.thisN']):
            plot_df_1['trial_number'] = trial['trial.thisN']
        else:
            print(f"Warning: 'trial.thisN' not found or contains invalid values for trial at index {index}")
        
        plot_df_1['time_norm'] = plot_df_1['time'] - trial['et_decision_start']
        
        # Add 'RT1', 'RT2', 'RT3' columns to plot_df_1
        for category_num in range(1, 4):
            category_choice_time_col = f'RT{category_num}'
            category_choice_time = all_subj.at[index, category_choice_time_col]
            plot_df_1[category_choice_time_col] = category_choice_time
        
        plot_df_1 = plot_df_1.dropna().reset_index(drop=True)
        
        total_entries = len(plot_df_1)
        
        if total_entries > 0:
            # Count the number of visits to Category 1 between RT1 and RT1 + RT2
            visits_category_1 = plot_df_1[
                (plot_df_1['time_norm'] >= plot_df_1['RT1']) &
                (plot_df_1['time_norm'] <= (plot_df_1['RT1'] + plot_df_1['RT2'])) &
                (plot_df_1['category'] == 'Category 1')
            ]
            
            visit_fraction = len(visits_category_1) / total_entries
            print(f"Trial {trial['trial.thisN']}: Fraction of revisits to Category 1 between RT1 and RT1 + RT2: {visit_fraction}")
            
            # Add a new row to the results DataFrame
            results_df = results_df.append({
                'Subject': subject_id,
                'Trial': trial['trial.thisN'],
                'Fraction_Revisits_Category1': visit_fraction
            }, ignore_index=True)

            appended_data_revisits.append(plot_df_1)

    return results_df

# Assuming participant is defined as a list of subjects
#participant = [1, 2, 3,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
participant=[1]
# Combine the results for all participants into a single DataFrame
all_results_df = pd.concat([revisit_category1_ategory_2_itsturn(subj_id) for subj_id in participant], ignore_index=True)

# Save the combined results to a CSV file
all_results_df.to_csv('revisit_category1_ategory_2_itsturn_all_subjects_results.csv', index=False)

# Combine the appended data into a DataFrame
revisits_data = pd.concat(appended_data_revisits, ignore_index=True)

#determinationOfthe percentage of lost eyedata
def data_lose():
    total_samples = len(etData)
    missing_samples = etData[etData['lr_x'].isna() | etData['lr_y'].isna()]
    num_missing_samples = len(missing_samples)
    data_loss = (num_missing_samples / total_samples) * 100
    print(f'Data loss: {data_loss:.2f}%')
data_lose()

#allsubject number of visit to category 1 while choosing category 2
def revisit_category1_ategory_2_itsturn(subject_id):
    global appended_data_revisits
    
    appended_data_revisits = []
    
    subj_data = all_subj[all_subj['participant'] == subject_id]
    
    for index, trial in subj_data.iterrows():
        gazedata = select_eyedata(trial, index, etData)
        
        translation = [int(trial.decision1_1category), int(trial.decision1_2category), int(trial.decision1_3category),
                       int(trial.decision2_1category), int(trial.decision2_2category), int(trial.decision2_3category),
                       int(trial.decision3_1category), int(trial.decision3_2category), int(trial.decision3_3category)]
        
        gazedata['graph'] = gazedata['roi'].map(dict(zip(translation, reversed(range(1, 10)))))
        
        # Create a new column 'category' based on the 'graph' column
        gazedata['category'] = gazedata['graph'].map({
            9: 'Category 1',
            8: 'Category 1',
            7: 'Category 1',
            6: 'Category 2',
            5: 'Category 2',
            4: 'Category 2',
            3: 'Category 3',
            2: 'Category 3',
            1: 'Category 3',
        })
        
        plot_df_1 = pd.DataFrame({'lr_y': gazedata.lr_y,
                                  'lr_x': gazedata.lr_x,
                                  'time': gazedata.time,
                                  'roi': gazedata.roi,
                                  'graph': gazedata.graph,
                                  'category': gazedata.category})  # Include the 'category' column in plot_df_1
        
        # Check if 'trial.thisN' exists in trial data and contains valid values
        if 'trial.thisN' in trial and pd.notna(trial['trial.thisN']):
            plot_df_1['trial_number'] = trial['trial.thisN']
        else:
            print(f"Warning: 'trial.thisN' not found or contains invalid values for trial at index {index}")
        
        plot_df_1['time_norm'] = plot_df_1['time'] - trial['et_decision_start']
        
        # Add 'RT1', 'RT2', 'RT3' columns to plot_df_1
        for category_num in range(1, 4):
            category_choice_time_col = f'RT{category_num}'
            category_choice_time = all_subj.at[index, category_choice_time_col]
            plot_df_1[category_choice_time_col] = category_choice_time
        
        plot_df_1 = plot_df_1.dropna().reset_index(drop=True)
        
        total_entries = len(plot_df_1)
        
        if total_entries > 0:
            # Count the number of visits to Category 1 between RT1 and RT1 + RT2
            visits_category_1 = plot_df_1[
                (plot_df_1['time_norm'] >= plot_df_1['RT1']) &
                (plot_df_1['time_norm'] <= (plot_df_1['RT1'] + plot_df_1['RT2'])) &
                (plot_df_1['category'] == 'Category 1')
            ]
            
            visit_fraction = len(visits_category_1) / total_entries
            print(f"Trial {trial['trial.thisN']}: Fraction of revisits to Category 1 between RT1 and RT1 + RT2: {visit_fraction}")
            
            # Add a new column 'fraction_revisits_category1' to plot_df_1
            plot_df_1['fraction_revisits_category1'] = visit_fraction
        
            appended_data_revisits.append(plot_df_1)
            
            # Save the plot for each trial
            plt.scatter(plot_df_1['lr_x'], plot_df_1['lr_y'], c=plot_df_1['fraction_revisits_category1'], cmap='viridis')
            plt.colorbar(label='Fraction of Revisits to Category 1')
            plt.xlabel('Gaze X Coordinate')
            plt.ylabel('Gaze Y Coordinate')
            plt.title(f'Revisit Category 1 (Subject {subject_id}, Trial {trial["trial.thisN"]})')
            #plt.savefig(f'revisit_category1_ategory_2_itsturn_subject_{subject_id}_trial_{trial["trial.thisN"]}.png', dpi=300, bbox_inches='tight', transparent=True)
            plt.clf()
        else:
            print(f"Trial {trial['trial.thisN']}: No valid entries for calculating fraction of revisits to Category 1.")

participant=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
#participant=[29]
# Call the function for all participants
for subj_id in participant:
    revisit_category1_ategory_2_itsturn(subj_id)

# Combine the appended data into a DataFrame
revisits_data = pd.concat(appended_data_revisits, ignore_index=True)

def add_label_band(ax, top, bottom, label, *, spine_pos=-0.05, tip_pos=-0.02):
    """
    Helper function to add bracket around y-tick labels.

    Parameters
    ----------
    ax : matplotlib.Axes
        The axes to add the bracket to

    top, bottom : floats
        The positions in *data* space to bracket on the y-axis

    label : str
        The label to add to the bracket

    spine_pos, tip_pos : float, optional
        The position in *axes fraction* of the spine and tips of the bracket.
        These will typically be negative

    Returns
    -------
    bracket : matplotlib.patches.PathPatch
        The "bracket" Aritst.  Modify this Artist to change the color etc of
        the bracket from the defaults.

    txt : matplotlib.text.Text
        The label Artist.  Modify this to change the color etc of the label
        from the defaults.

    """
    # grab the yaxis blended transform
    transform = ax.get_yaxis_transform()

    # add the bracket
    bracket = mpatches.PathPatch(
        mpath.Path(
            [
                [tip_pos, top],
                [spine_pos, top],
                [spine_pos, bottom],
                [tip_pos, bottom],
            ]
        ),
        transform=transform,
        clip_on=False,
        facecolor="none",
        edgecolor="k",
        linewidth=1,
    )
    ax.add_artist(bracket)

    # add the label
    txt = ax.text(
        spine_pos,
        (top + bottom) / 2,
        label,
        ha="right",
        va="center",
        rotation="vertical",
        clip_on=False,
        transform=transform,
    )

    return bracket, txt

def add_label_band(ax, top, bottom, label, *, spine_pos=-0.05, tip_pos=-0.02):
    """
    Helper function to add bracket around y-tick labels.

    Parameters
    ----------
    ax : matplotlib.Axes
        The axes to add the bracket to

    top, bottom : floats
        The positions in *data* space to bracket on the y-axis

    label : str
        The label to add to the bracket

    spine_pos, tip_pos : float, optional
        The position in *axes fraction* of the spine and tips of the bracket.
        These will typically be negative

    Returns
    -------
    bracket : matplotlib.patches.PathPatch
        The "bracket" Artist.  Modify this Artist to change the color etc of
        the bracket from the defaults.

    txt : matplotlib.text.Text
        The label Artist.  Modify this to change the color etc of the label
        from the defaults.

    """
    # grab the y-axis blended transform
    transform = ax.get_yaxis_transform()

    # add the bracket
    bracket = mpatches.PathPatch(
        mpath.Path(
            [
                [tip_pos, top],
                [spine_pos, top],
                [spine_pos, bottom],
                [tip_pos, bottom],
            ]
        ),
        transform=transform,
        clip_on=False,
        facecolor="none",
        edgecolor="k",
        linewidth=1,
    )
    ax.add_artist(bracket)

    # add the label with Times New Roman font and size 16
    txt = ax.text(
        spine_pos,
        (top + bottom) / 2,
        label,
        ha="right",
        va="center",
        rotation="vertical",
        clip_on=False,
        transform=transform,
        fontname="Times New Roman",
        fontsize=16
    )

    return bracket, txt

def gaze_plt():
    global appended_data_gaze_all
    appended_data_gaze_all = []

    for index, trial in all_subj.iterrows():
        # select eyetracking data for the trial:
        gazedata = select_eyedata(trial, index, etData)

        # Vector with the positions of images ordered based on choice order of the trial
        translation = [
            int(trial.decision1_1category), int(trial.decision1_2category),
            int(trial.decision1_3category), int(trial.decision2_1category),
            int(trial.decision2_2category), int(trial.decision2_3category),
            int(trial.decision3_1category), int(trial.decision3_2category),
            int(trial.decision3_3category)
        ]

        # The first selected option of the first category should be on top of the graph so it becomes number 9 and so on
        gazedata.loc[gazedata['roi'] == translation[0], 'graph'] = 9
        gazedata.loc[gazedata['roi'] == translation[1], 'graph'] = 8
        gazedata.loc[gazedata['roi'] == translation[2], 'graph'] = 7
        gazedata.loc[gazedata['roi'] == translation[3], 'graph'] = 6
        gazedata.loc[gazedata['roi'] == translation[4], 'graph'] = 5
        gazedata.loc[gazedata['roi'] == translation[5], 'graph'] = 4
        gazedata.loc[gazedata['roi'] == translation[6], 'graph'] = 3
        gazedata.loc[gazedata['roi'] == translation[7], 'graph'] = 2
        gazedata.loc[gazedata['roi'] == translation[8], 'graph'] = 1

        # DataFrame with time, looked at location, and trial number
        plot_df = pd.DataFrame({
            'graph': gazedata['graph'],
            'lr_x': gazedata['lr_x'],
            'lr_y': gazedata['lr_y'],
            'time': gazedata['time'],
            'trial': trial['trial.thisN']  # Add trial number column
        })
        plot_df = plot_df.dropna()
        plot_df = plot_df.reset_index(drop=True)
        appended_data_gaze_all.append(plot_df)

        plt.figure()

        # Plot each trial on the same figure with labels
        plt.plot(
            plot_df['time'] - trial['et_decision_start'],
            plot_df['graph'],
            '-',
            linewidth="0.5",
            color="grey"
        )
        plt.plot(
            plot_df['time'] - trial['et_decision_start'],
            plot_df['graph'],
            '.',
            markersize=2,
            color="black"
        )

        # Add axhspan with transparency
        plt.axhspan(6.5, 9.5, facecolor='coral', alpha=0.45, edgecolor='coral')
        plt.axhspan(3.5, 6.5, facecolor='orangered', alpha=0.6, edgecolor='orangered')
        plt.axhspan(0, 3.5, facecolor='lightsalmon', alpha=0.25, edgecolor='lightsalmon')
        # Plot chosen markers
        plt.plot(trial['decisiontime1'], 9, marker="o", markersize=12,
                 markerfacecolor='indigo', markeredgecolor='indigo')
        plt.plot(trial['decisiontime2'], 6, marker="o", markersize=12,
                 markerfacecolor='olive', markeredgecolor='olive')
        plt.plot(trial['decisiontimetotal'], 3, marker="o", markersize=12,
                 markerfacecolor='lightblue', markeredgecolor='lightblue')

        # Plot y-axis tick labels with specified colors and font properties
        ytick_labels = ['Unchosen', 'Unchosen', 'Chosen 3', 'Unchosen', 'Unchosen', 'Chosen 2', 'Unchosen', 'Unchosen', 'Chosen 1']
        colors = ['black', 'black', 'lightblue', 'black', 'black', 'olive', 'black', 'black', 'indigo']
        font = {'family': 'Times New Roman',  'size': 14}
        for i, label in enumerate(ytick_labels):
            font_weight = 'bold' if 'Chosen' in label else 'normal'
            plt.text(0, i+1, label, color=colors[i], fontdict={'family': 'Times New Roman', 'size': 16, 'weight': font_weight}, ha='right', va='center')

        # Add label for y-axis
        plt.text(-0.1, 1.02, 'Options', color='black', fontdict={'family': 'Times New Roman', 'size': 18}, transform=plt.gca().transAxes, ha='center', va='bottom')
        #plt.text(0.5, 1.02, 'Options', color='black', fontdict={'family': 'Times New Roman', 'size': 18}, transform=plt.gca().transAxes, ha='center', va='bottom')
        plt.ylabel('Chosen Category Order', fontdict={'family': 'Times New Roman', 'size': 18}, labelpad=80)

        # Add brackets
        add_label_band(plt.gca(), 6.75, 9.4,'',spine_pos=-0.2, tip_pos=-0.18)
        add_label_band(plt.gca(), 3.8, 6.4,'',spine_pos=-0.2, tip_pos=-0.18)
        add_label_band(plt.gca(), 0.75, 3.4,'',spine_pos=-0.2, tip_pos=-0.18)

        # Other plot configurations
        #plt.arrow(-5,0, 0, 1, color='black', linewidth=5)
        plt.axhline(y=-0.5, color='black', linewidth=1)
        plt.ylim([0.5, 9.5])
        plt.xlabel('Time (s)', fontdict={'family': 'Times New Roman', 'size': 18})
        plt.tick_params(axis='x', labelsize=18)
        plt.xlim(0, (trial['et_decision_end'] + 1) - trial['et_decision_start'])
        plt.grid(False)
        plt.yticks([])  # Removing y-axis ticks
        plt.savefig(f'gazeplot_trial_{trial["trial.thisN"]}.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.show()

# Call the function
gaze_plt()
#BoxPlotOfTheDistributionOfReactionTime
plot_size = (18, 4)

def average_choice_time_distribution():

    global data
    global participant_anova

    # Assuming results, RT1, RT2, and RT3 are defined

    # change the format to long instead of wide to be able to use the seaborn library for data visualization
    data = pd.melt(results[['RT1', 'RT2', 'RT3']])

    # plot mean importance for category 1,2,3
    imp = plt.figure()
    imp = sns.boxplot(x="variable", y="value", data=data, palette=["indigo", "olive", "lightblue"])
    plt.xticks([0, 1, 2], ['RT1', 'RT2', 'RT3'], fontsize=18, fontname="Times New Roman")
    plt.xlabel('')
    plt.yticks(fontsize=18, fontname="Times New Roman")  # Set y-axis font size and font name

    # Remove x-axis label
    plt.xlabel('')
    plt.ylabel("Time (s)", fontsize=18, fontname="Times New Roman")
    #plt.ylim()
    add_stat_annotation(imp, data=data, x="variable", y="value",  # add statistical annotations to the plot
                        box_pairs=[('RT1', 'RT2'), ('RT1',
                                                     'RT3'), ('RT3', 'RT2')],
                        test='t-test_paired', text_format='star', loc='inside', verbose=2, fontsize=18)
    # Customize y-axis ticks and labels
    # Customize y-axis ticks and labels
    #y_ticks = ax.get_yticks()
    
    #plt.axvline(x=-0.5, color='black', linewidth=3)
#Add small horizontal lines at each y-axis tick
   # plt.grid(False)

    #for y in y_ticks:
    #    ax.plot([-5, -4.5], [y, y], color='k', linewidth=3)

#plt.axhline(y=-0.9, linewidth=1, color='black', zorder=1)
#plt.axvline(x=-0.9, linewidth=1, color='black', zorder=1)
# Add vertical line
    # Set y-axis ticks at 5-step intervals
    # Set y-axis ticks at 5-step intervals
    y_ticks = np.arange(0, plt.gca().get_ylim()[1], 5)
    plt.yticks(y_ticks, [str(int(y)) for y in y_ticks], fontsize=18)

    # Add small horizontal lines at each y-axis tick
    for y in y_ticks:
        plt.plot([-0.5, -0.45], [y, y], color='k', linewidth=2)

    # Add vertical line at x=-0.5
    plt.axvline(x=-0.5, color='k', linewidth=3)
    # Add horizontal line at y=0 (x-axis line)
    plt.axhline(y=0, color='k', linewidth=3)


    # Remove grid lines
    plt.grid(False)


    imp.figure.savefig("RT.png", format="png", dpi=900,
                       transparent=True)  # save figure

average_choice_time_distribution()