import os
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing 
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from flask import session
import pickle
import io
import base64
import mpld3
import ast
import io
import base64

def visual(df,bacts,d_type,plot_type,conc,vol,slide,trails):
    #bacts= ast.literal_eval(bacts)
    data = df[df['bacteria'].isin(bacts)] 
    print(data['bacteria'].unique())
    # Initialize a figure and subplot layout
    unique_bacteria  = data['bacteria'].unique()
    fol = "Plots/cluster_plots"
    if d_type == "raw_data":
        data=data[(data['concentration']==float(conc)) & (data['volume']==float(vol))& (data['slide']==slide)]
        data['mag'] = np.sqrt(data['Ypos']**2 + data['Xpos']**2)/ data['Pow']
        data['dir'] = np.arctan2(data['Ypos'], data['Xpos'])
        if len(trails)>0:
            print("entered Trails")
            data = data[data['trail'].isin(trails)] 
            print(data.head())    
        else:
            if len(data['trail'].unique())>1:
                print("averaging trials")
                print("entered AVG")
                data_trails = data
                #data = avg_calculator(data)
                print(data.head())
        fol="Plots/raw_plots"
    print(data.head())
    print(data['bacteria'].unique())
    if not os.path.exists(fol):
        os.mkdir(fol)
    if len(trails)>0:
        print("printing Trails")
        unq_trails = data['trail'].unique()
        if not os.path.exists(fol+"/Merged_plots"):
            fol = fol+"/Merged_plots"
            os.mkdir(fol)
        plt.figure(figsize=(10, 6))
        # Iterate over unique bacteria values and plot corresponding data
        for i, trail in enumerate(unq_trails):
            # Filter data for the current bacteria
            data_for_bacteria = data[data['trail'] == trail] 
            #data_for_bacteria2 = data_trails[data_trails['bacteria']==bacteria]
            # Plot data for the current bacteria
            plt.plot(data_for_bacteria['mag'], data_for_bacteria['dir'], label=trail)
            #plt.plot(data_for_bacteria2['mag'], data_for_bacteria2['dir'], label=bacteria)
        # Add labels and legend
        plt.xlabel("mag")
        plt.ylabel("Dir")
        plt.title(unique_bacteria[0])
        plt.legend()
        img_bytes_io = io.BytesIO()
        plt.savefig(img_bytes_io)
        f_name = fol+'/'+"trails"+unique_bacteria[0]+'_'+d_type+".png"
        plt.savefig(f_name)
        img_bytes_io.seek(0)
        img_base64 = base64.b64encode(img_bytes_io.read()).decode('utf-8')
        plt.close()  # Close the plot to free up resources
        return img_base64

    if plot_type == "Indi":
        if not os.path.exists(fol+"/Individual_plots"):
            fol = fol+"/Individual_plots"
            os.mkdir(fol)
        print("enterd Indi")
        fig, axs = plt.subplots(nrows=len(unique_bacteria), figsize=(10, 6))
        for i, bacteria in enumerate(unique_bacteria):
            data_for_bacteria = data[data['bacteria'] == bacteria]
            axs[i].plot(data_for_bacteria['mag'], data_for_bacteria['dir'], label=bacteria)
            axs[i].set_title(bacteria)
            axs[i].set_xlabel("Mag")
            axs[i].set_ylabel("Dir")
            axs[i].legend()
            f_name = fol+'/'+bacteria+'_'+d_type+".png"
            axs[i].figure.savefig(f_name)
        print("exitted for")
        plt.tight_layout()
        img_bytes_io = io.BytesIO()
        plt.savefig(img_bytes_io)
        img_bytes_io.seek(0)
        img_base64 = base64.b64encode(img_bytes_io.read()).decode('utf-8')
        plt.close()  # Close the plot to free up resources
        return img_base64
    else:
        print("plotting AVG")
        if not os.path.exists(fol+"/Merged_plots"):
            fol = fol+"/Merged_plots"
            os.mkdir(fol)
        plt.figure(figsize=(10, 6))
        # Iterate over unique bacteria values and plot corresponding data
        for i, bacteria in enumerate(unique_bacteria):
            # Filter data for the current bacteria
            data_for_bacteria = data[data['bacteria'] == bacteria] 
            #data_for_bacteria2 = data_trails[data_trails['bacteria']==bacteria]
            # Plot data for the current bacteria
            plt.plot(data_for_bacteria['mag'], data_for_bacteria['dir'], label=bacteria)
            #plt.plot(data_for_bacteria2['mag'], data_for_bacteria2['dir'], label=bacteria)
        # Add labels and legend
        plt.xlabel("Mag")
        plt.ylabel("Dir")
        plt.title("Bacteria Plot")
        plt.legend()
        img_bytes_io = io.BytesIO()
        plt.savefig(img_bytes_io)
        f_name = fol+'/'+bacteria+'_'+d_type+".png"
        plt.savefig(f_name)
        img_bytes_io.seek(0)
        img_base64 = base64.b64encode(img_bytes_io.read()).decode('utf-8')
        plt.close()  # Close the plot to free up resources
        return img_base64
    
def avg_calculator(data):
    columns = ['mag', 'dir','Pow','bacteria']
    newD = pd.DataFrame(columns=columns)
    for i in data['bacteria'].unique():
        columns = ['mag', 'dir','Pow']
        empty_dataset = pd.DataFrame(columns=columns)
        dataX=data[data['bacteria']==i]
        data1 = dataX[dataX["trail"]=="T1"]
        data2 = dataX[dataX["trail"]=="T2"]
        data3 = dataX[dataX["trail"]=="T3"]
        empty_dataset['mag'] =  (np.array(data1['mag'])+np.array(data2['mag'])+np.array(data3['mag']))/3
        empty_dataset['dir'] = (np.array(data1['dir'])+np.array(data2['dir'])+np.array(data3['dir']))/3
        empty_dataset['Pow'] = (np.array(data1['Pow'])+np.array(data2['Pow'])+np.array(data3['Pow']))/3
        empty_dataset['bacteria'] = i
        newD=pd.concat([newD,empty_dataset])
    return newD

def clustered_plotter(df,bacts):
    data = df[df['bacteria'].isin(bacts)] 
    print(data['bacteria'].unique())
    # Initialize a figure and subplot layout
    unique_bacteria  = data['bacteria'].unique()
    fol = "Plots/cluster_plots"
    print("plotting clustred_data")
    if not os.path.exists(fol+"/clustered_plots"):
        fol = fol+"/clustered_plots"
        os.mkdir(fol)
    plt.figure(figsize=(10, 6))
    # Iterate over unique bacteria values and plot corresponding data
    for i, bacteria in enumerate(unique_bacteria):
        # Filter data for the current bacteria
        data_for_bacteria = data[data['bacteria'] == bacteria] 
        #data_for_bacteria2 = data_trails[data_trails['bacteria']==bacteria]
        # Plot data for the current bacteria
        plt.plot(data_for_bacteria['mag'], data_for_bacteria['dir'], label=bacteria)
        #plt.plot(data_for_bacteria2['mag'], data_for_bacteria2['dir'], label=bacteria)
    # Add labels and legend
    plt.xlabel("Mag")
    plt.ylabel("Dir")
    plt.title("Bacteria Plot")
    plt.legend()
    img_bytes_io = io.BytesIO()
    plt.savefig(img_bytes_io)
    f_name = fol+'/'+bacteria+'_'+'cluster'+".png"
    plt.savefig(f_name)
    img_bytes_io.seek(0)
    img_base64 = base64.b64encode(img_bytes_io.read()).decode('utf-8')
    plt.close()  # Close the plot to free up resources
    return img_base64




