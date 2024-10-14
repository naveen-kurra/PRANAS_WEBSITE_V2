import scipy.stats as stats 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io,base64
import pandas as pd
bacteria_colors = {'EC': 'blue', 'SA': 'green', 'PA': 'red', 'KP': 'black', 'SM': 'purple', 'SE': 'brown','ECSA':'orange','PASE':'cyan','SMKP':'teal','LM':'pink'}
def calculate_sa(group):
    max_intensity = group['Pow'].max()
    min_intensity = group['Pow'].min()
    mean_intensity = group['Pow'].mean()
    
    # Calculate SA
    if mean_intensity != 0:
        return (max_intensity - min_intensity) / mean_intensity
    else:
        return np.nan
def features_creator(averaged_data):
    print(averaged_data.head())
    averaged_data['Mag'] = np.sqrt(averaged_data['Ypos']**2 + averaged_data['Xpos']**2)
    averaged_data['BDE'] =averaged_data['Mag'] / averaged_data['Pow']
    averaged_data['dir'] = np.arctan2(averaged_data['Ypos'], averaged_data['Xpos'])
    averaged_data['SIV'] = averaged_data.groupby(['bacteria'])['Pow'].transform(np.std)
    center_x, center_y = averaged_data['Xpos'].mean(), averaged_data['Ypos'].mean()
    averaged_data['SCD'] = np.sqrt((averaged_data['Xpos'] - center_x) ** 2 + (averaged_data['Ypos'] - center_y) ** 2)
    averaged_data['AoS'] = np.arctan2(averaged_data['Ypos'] - center_y, averaged_data['Xpos'] - center_x)
    psi_values = averaged_data.groupby('bacteria')['Pow'].max()
    averaged_data['PSI'] = averaged_data['bacteria'].map(psi_values)
    msa_values = averaged_data.groupby('bacteria')['dir'].mean()
    averaged_data['MSA'] = averaged_data['bacteria'].map(msa_values)
    tsp_values = averaged_data.groupby('bacteria')['Pow'].sum()
    averaged_data['TSP'] = averaged_data['bacteria'].map(tsp_values)
    ssi_values = averaged_data.groupby('bacteria').apply(lambda x: abs(x['Pow'] - x['Pow'].iloc[::-1]).mean())
    averaged_data['SSI'] = averaged_data['bacteria'].map(ssi_values)
    se_values = averaged_data.groupby('bacteria')['Pow'].apply(lambda x: stats.entropy(x))
    averaged_data['SE'] = averaged_data['bacteria'].map(se_values)
    sa_values = averaged_data.groupby('bacteria').apply(calculate_sa)
    averaged_data['SA'] = averaged_data['bacteria'].map(sa_values)
    return averaged_data
def data_split_group(data,bact):
    data1=data[data['bacteria'].isin(bact)]
    return data1
def determine_d_t(bacteria_list):
    if all(bacteria in ["SA", "SE", "SM"] for bacteria in bacteria_list) and len(bacteria_list) > 0:
        d_t = "gp_data"
    elif all(bacteria in ["EC", "PA", "KP"] for bacteria in bacteria_list) and len(bacteria_list) > 0:
        d_t = "gn_data"
    elif all(bacteria in ["ECSA", "SMKP", "PASE"] for bacteria in bacteria_list) and len(bacteria_list) > 0:
        d_t = "com_data"
    else:
        d_t = "cus_data"
    return d_t


def avg_calculator(data):
    columns = ['Xpos','Ypos','Pow','bacteria','slide']
    newD = pd.DataFrame(columns=columns)
    for i in data['bacteria'].unique():
        columns = ['Xpos','Ypos','Pow','slide']
        empty_dataset = pd.DataFrame(columns=columns)
        dataX=data[data['bacteria']==i]
        data1 = dataX[dataX["trail"]=="T1"]
        data2 = dataX[dataX["trail"]=="T2"]
        data3 = dataX[dataX["trail"]=="T3"]
        #empty_dataset['mag'] =  (np.array(data1['mag'])+np.array(data2['mag'])+np.array(data3['mag']))/3
        #empty_dataset['dir'] = (np.array(data1['dir'])+np.array(data2['dir'])+np.array(data3['dir']))/3
        empty_dataset['Xpos']=(np.array(data1['Xpos'])+np.array(data2['Xpos'])+np.array(data3['Xpos']))/3
        empty_dataset['Ypos']=(np.array(data1['Ypos'])+np.array(data2['Ypos'])+np.array(data3['Ypos']))/3
        empty_dataset['Pow']=(np.array(data1['Pow'])+np.array(data2['Pow'])+np.array(data3['Pow']))/3
        empty_dataset['bacteria'] = i
        empty_dataset['slide'] = data1['slide']
        newD=pd.concat([newD,empty_dataset])
    return newD

plt.ylabel('dir')
    # Show plot
plt.show()

def main_function_visual(df,bacts,conc,vol,slide,trails):
    fol="plots/raw_plots"
    data_type=determine_d_t(bacts)
    print(data_type)
    conc=float(conc)
    #vol=float(vol)
    vol = list(map(int, vol))
    print("O data is: ",df.head())
    df=df[(df['concentration']==conc)]
    df=df[(df['volume'].isin(vol))]
    print("J data is: ",df.head())
    f_data=features_creator(df)
    data_using=data_split_group(f_data,bacts)
    #print(data_using.head())
    plt.figure(figsize=(10, 6))
    data_using = data_using[data_using['slide'].isin(slide)]
    #print(data_using.head())
    sns.scatterplot(data=data_using, x='BDE', y='dir', hue='bacteria',style='slide', palette=bacteria_colors)
    #plt.title('BDE vs dir with hue as Bacteria and palette as Slide')
    plt.xlabel('BDE', fontsize=16)
    plt.ylabel('DIR', fontsize=16)
    plt.legend(title='Bacteria',  loc='best')
    plt.grid('on')
    img_bytes_io = io.BytesIO()
    plt.savefig(img_bytes_io)
    my_string = '_'.join(slide)
    f_name = fol+'/'+data_type+my_string+'raw'+".png"
    plt.savefig(f_name)
    img_bytes_io.seek(0)
    img_base64 = base64.b64encode(img_bytes_io.read()).decode('utf-8')
    plt.close()  # Close the plot to free up resources
    return img_base64
    #plt.show()