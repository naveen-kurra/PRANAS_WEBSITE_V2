import scipy.stats as stats 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io,base64
import pandas as pd
import scipy.stats as stats 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io,base64
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io,base64
import pandas as pd
import os,pickle
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
def PCA_1(gn_data,scaler):
    #gn_data = gn_data[(gn_data['slide'].isin())]
    
    features_for_pca = ['SIV','BDE','dir']
    X_pca = gn_data[features_for_pca]
    unique_values_string = '_'.join(gn_data['bacteria'].unique())
    X_pca_scaled = scaler.fit_transform(X_pca)
    if not os.path.exists("StandardScalers/"):
        os.mkdir("StandardScalers/")
    scaler_name= "StandardScalers/"+unique_values_string+'_scaler.pkl'
    with open(scaler_name, 'wb') as file:
        pickle.dump(scaler, file)
    # Apply PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_pca_scaled)
    if not os.path.exists("PCA_models/"):
        os.mkdir("PCA_models/")
    pca_name= "PCA_models/"+unique_values_string+'_pca.pkl'
    with open(pca_name, 'wb') as file:
        pickle.dump(pca, file)
    # Add PCA components to the data
    gn_data['PCA1'], gn_data['PCA2'] = pca_components[:, 0], pca_components[:, 1]

    # Calculate centroids for each bacteria type in the PCA space
    centroids = gn_data.groupby('bacteria')[['PCA1', 'PCA2']].mean()

    # Calculate Euclidean distance of each point from its corresponding bacteria centroid
    gn_data['distance_from_centroid'] = gn_data.apply(
        lambda row: np.linalg.norm(row[['PCA1', 'PCA2']] - centroids.loc[row['bacteria']]),
        axis=1
    )

    # Define a distance threshold - for this example, we'll use the mean distance plus one standard deviation
    distance_threshold = gn_data['distance_from_centroid'].mean() + gn_data['distance_from_centroid'].std()

    # Filter the data to retain only points within the threshold distance from the centroid
    filtered_data = gn_data[gn_data['distance_from_centroid'] <= distance_threshold]
    return filtered_data,scaler,pca
def main_function_pca(df,bacts,conc,vol,slide,trails):
    fol="plots/pca_plots"
    data_type=determine_d_t(bacts)
    print(data_type)
    conc=float(conc)
    vol=float(vol)
    df=df[(df['concentration']==conc)&(df['volume']==vol)]
    f_data=features_creator(df)
    data_using=data_split_group(f_data,bacts)
    plt.figure(figsize=(10, 6))
    data_using = data_using[data_using['slide'].isin(slide)]
    scaler1 = StandardScaler()
    filtered_data,scaler2,pca2 = PCA_1(data_using,scaler1)
    # Visualize the filtered PCA data
    plt.figure(figsize=(14, 8))
    sns.scatterplot(data=filtered_data, x='PCA1', y='PCA2', hue='bacteria', style='slide',palette=bacteria_colors)
    # plt.title('Filtered PCA of Bacteria Data Close to Centroids')
    plt.xlabel('PCA Component 1', fontsize=16)
    plt.ylabel('PCA Component 2', fontsize=16)
    plt.grid(True)
    img_bytes_io = io.BytesIO()
    plt.savefig(img_bytes_io)
    my_string = '_'.join(slide)
    f_name = fol+'/'+data_type+my_string+'pca'+".png"
    plt.savefig(f_name)
    img_bytes_io.seek(0)
    img_base64 = base64.b64encode(img_bytes_io.read()).decode('utf-8')
    plt.close()  # Close the plot to free up resources
    return img_base64