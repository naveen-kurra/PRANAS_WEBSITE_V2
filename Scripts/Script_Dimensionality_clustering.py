# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 09:32:50 2024

@author: kurra
"""

from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import warnings
import random
from datetime import datetime
warnings.filterwarnings('ignore')
import os
import ast
import io
import base64
import matplotlib.pyplot as plt

def Clustered_data(df,algorithm,bacts,conc,vol,slide):
    bacts= ast.literal_eval(bacts)
    data = df[df['bacteria'].isin(bacts)]
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    inp=algorithm
    data=data[(data['concentration']==float(conc)) & (data['volume']==float(vol))& (data['slide']==slide)]
    data_3bact=data
    data_3bact['mag'] = np.sqrt(data_3bact['Ypos']**2 + data_3bact['Xpos']**2) / data_3bact['Pow']
    #data_3bact['mag'] = np.sqrt(data_3bact['Ypos']**2 + data_3bact['Xpos']**2)
    data_3bact['dir'] = np.arctan2(data_3bact['Ypos'], data_3bact['Xpos'])
    print(data_3bact.head())
    print("Concentraion is : ",data_3bact['concentration'].unique(),'/n', "Volume is : ",data_3bact['volume'].unique(),'/n', "Slide is : ",data_3bact['slide'].unique())
    bact = data_3bact['bacteria'].unique()
    #data_3bact = avg_calculator(data_3bact)
    current_date = datetime.now().strftime("%Y_%m_%d")
    if not os.path.exists('Clustered_data'):
            os.makedirs('Clustered_data')
    if inp=='dbs':
        for i in range(len(bact)):
            print(bact[i])
            data_3bact2 =data_3bact[data_3bact['bacteria']==bact[i]]
            data_3bact_processed = data_3bact2[["mag", "dir"]].values
            #data_3bact_processed = data_3bact2[["Ypos", "Pow"]].values


            n_clusters = 3


            # Initialize and fit the K-Means model
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(data_3bact_processed)
            unique_clusters = np.unique(cluster_labels)
            # Calculate silhouette scores for each sample
            silhouette_avg = silhouette_score(data_3bact_processed, cluster_labels)

            # Find the best cluster (cluster with highest silhouette score)
            best_cluster_idx = silhouette_avg.argmax()

            # Extract data for the best cluster
            best_cluster_mask = cluster_labels == best_cluster_idx
            best_cluster_data = data_3bact_processed[best_cluster_mask].copy()
            best_cluster_data = pd.DataFrame(best_cluster_data, columns=['mag', 'dir'])
            #best_cluster_data = pd.DataFrame(best_cluster_data, columns=['YPos', 'Pow'])
            # Add an additional column to the best cluster data
            best_cluster_data['bacteria'] = bact[i]
            best_cluster_data['slide'] = slide
            # Save the best cluster data as a CSV file
            filename_BC='best_cluster_data_DBSCAN_'+bact[i]+'.csv'
            best_cluster_data.to_csv(filename_BC, index=False)
    elif inp=='KMS':
        n_clusters = 3
        data_label = "Clustered_data/"+inp+"/clustered_data_Kmeans_"+current_date 
        if not os.path.exists(data_label):
                os.makedirs(data_label, exist_ok=True)
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        print("Kmeans Started at : ", formatted_datetime)
        for i in range(len(bact)):
            print(bact[i])
            data_3bact2 =data_3bact[data_3bact['bacteria']==bact[i]]
            data_3bact_processed = data_3bact2[["mag", "dir"]].values
            #data_3bact_processed = data_3bact2[["Pow", "Ypos"]].values
            n_clusters = 3
            print("KMeans Started")
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            kmeans.fit(data_3bact_processed)
            print("KMeans Completed")
            cluster_labels = kmeans.labels_
            print("silhouette_score Started")
            silhouette_avg = silhouette_score(data_3bact_processed, cluster_labels)
            best_cluster_idx = silhouette_avg.argmax()
            print("silhouette_score completed")
            best_cluster_mask = cluster_labels == best_cluster_idx
            best_cluster_data = data_3bact_processed[best_cluster_mask].copy()
            best_cluster_data = pd.DataFrame(best_cluster_data, columns=['mag', 'dir'])
            best_cluster_data['bacteria'] = bact[i]
            conc = conc
            vol = vol
            filename_BC=data_label+'/'+'best_cluster_data_Kmeans_'+str(conc)+'_'+str(vol)+'_'+bact[i]+'.csv'
            best_cluster_data.to_csv(filename_BC, index=False)     
        res = merge_csv(data_label,"KMS",slide)
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        print("Kmeans Ended at : ", formatted_datetime)
        print(res)
    elif inp=='OPS':
        n_clusters = 3
        data_label = "Clustered_data/"+inp+"/clustered_data_OPTCIS_"+current_date
        if not os.path.exists(data_label):
                os.makedirs(data_label, exist_ok=True)
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        print("OPTICS Started at : ", formatted_datetime)
        for i in range(len(bact)):
            print(bact[i])
            data_3bact2 =data_3bact[data_3bact['bacteria']==bact[i]]
            data_3bact_processed = data_3bact2[["mag", "dir"]].values
            #data_3bact_processed = data_3bact2[["Pow", "Ypos"]].values
            n_clusters = 3
            optics = OPTICS(min_samples=10, max_eps=np.inf)
            print("OPTICS Started")
            cluster_labels = optics.fit_predict(data_3bact_processed)
            unique_clusters = np.unique(cluster_labels)
            print("OPTICS Completed")
            print("silhouette_score Started")
            silhouette_avg = silhouette_score(data_3bact_processed, cluster_labels)
            best_cluster_idx = unique_clusters[silhouette_avg.argmax()]
            print("silhouette_score completed")
            best_cluster_mask = cluster_labels == best_cluster_idx
            best_cluster_data = data_3bact_processed[best_cluster_mask].copy()
            best_cluster_data = pd.DataFrame(best_cluster_data, columns=['mag', 'dir'])
            best_cluster_data['bacteria'] = bact[i]
            conc = conc
            vol = vol
            filename_BC=data_label+'/'+'best_cluster_data_OPTCIS_'+str(conc)+'_'+str(vol)+'_'+bact[i]+'.csv'
            best_cluster_data.to_csv(filename_BC, index=False)     
        res = merge_csv(data_label,"OPS",slide)
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        print("OPTICS Ended at : ", formatted_datetime)
        print(res)
    else:
        print("please choose a valid algorithm to process further, Thanks :)")
    return "Thanks for using this method. Please find the saved files in this directory. :)"

def merge_csv(folder_path,algoSTR,slide):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Initialize an empty DataFrame to store the merged data
    merged_data = pd.DataFrame()

    # Loop through each CSV file and merge it into the main DataFrame
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        merged_data = pd.concat([merged_data, data], ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    f_name = datetime.now().strftime("%Y_%m_%d")
    f_name = 'Data_files/clustered_data/Clustered_'+algoSTR+'_'+slide+'_MasterData_'+f_name+'.csv'
    merged_data.to_csv(f_name, index=False)
    return "successful"


def avg_calculator(data):
    columns = ['mag', 'dir','bacteria']
    newD = pd.DataFrame(columns=columns)
    for i in data['bacteria'].unique():
        columns = ['mag', 'dir']
        empty_dataset = pd.DataFrame(columns=columns)
        dataX=data[data['bacteria']==i]
        data1 = dataX[dataX["trail"]=="T1"]
        data2 = dataX[dataX["trail"]=="T2"]
        data3 = dataX[dataX["trail"]=="T3"]
        empty_dataset['mag'] =  (np.array(data1['mag'])+np.array(data2['mag'])+np.array(data3['mag']))/3
        empty_dataset['dir'] = (np.array(data1['dir'])+np.array(data2['dir'])+np.array(data3['dir']))/3
        empty_dataset['bacteria'] = i
        newD=pd.concat([newD,empty_dataset])
    return newD