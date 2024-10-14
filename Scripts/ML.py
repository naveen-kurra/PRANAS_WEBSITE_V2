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
import pickle
import io
import base64
import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn import svm
from sklearn import preprocessing 
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression


# Training the model with optics data and saaving the plot of original point an
def Training_ML(df,meth,bacts):
    data = df
    print(bacts)
    print(data['bacteria'].unique())
    data = data[data['bacteria'].isin(bacts)] 
    print(data['bacteria'].unique())
    if meth == "GP":
    # Garm positive
       data_3bact = data[(data['bacteria']=='SA') | (data['bacteria']=='SE') | (data['bacteria']=='SM')]
    #Gram Negative
    elif meth == "GN":
        data_3bact = data[(data['bacteria']=='EC') | (data['bacteria']=='PA') | (data['bacteria']=='KP')]
    #Combinations
    elif meth == "COM":
        data_3bact = data[(data['bacteria']=='ECSA') | (data['bacteria']=='PASE') | (data['bacteria']=='SMKP')]
    else:
        meth = data['bacteria'].unique()
        meth = '_'.join(map(str, meth))
        
    data_3bact = data
    if 'mag' not in data.columns:
        data['mag'] = np.sqrt(data['Ypos']**2 + data['Xpos']**2) / data['Pow']
        data['dir'] = np.arctan2(data['Ypos'], data['Xpos'])
        
    X = data[['mag', 'dir']].values
    y = data['bacteria']
    print(data['bacteria'].unique())

        ########## ALL bacteria Strung generator ###########

    unique_bacteria = data['bacteria'].unique()
    unique_bacteria_strings = [str(bacteria) for bacteria in unique_bacteria]
    # Join the unique string values with underscores
    bacteria_string = "_".join(unique_bacteria_strings)

    colors = {'SA': 'red', 'SE': 'green', 'SM': 'blue','EC': 'orange', 'PA': 'pink', 'KP': 'Olive','ECSA': 'purple', 'PASE': 'cyan', 'SMKP': 'Magenta'}
        ####### Label Encoder ########
    
    label_encoder = preprocessing.LabelEncoder() 
    y_label= label_encoder.fit_transform(y) 
    # Save the scaler to a file
    if not os.path.exists("LabelEncoders/"):
        os.mkdir("LabelEncoders/")
    Le_name= "LabelEncoders/"+bacteria_string+'_Label_encoder.pkl'
    with open(Le_name, 'wb') as file:
        pickle.dump(label_encoder, file)

        ####### Standard Scaler ########

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    # Save the scaler to a file
    if not os.path.exists("StandardScalers/"):
        os.mkdir("StandardScalers/")
    scaler_name= "StandardScalers/"+bacteria_string+'_scaler.pkl'
    with open(scaler_name, 'wb') as file:
        pickle.dump(scaler, file)


    random_seed = 42

    ####### Model Fitting and Plots ########

    if not os.path.exists('Plots/ML_Plots'):
        os.mkdir('Plots/ML_Plots')
    logreg = LogisticRegression(random_state=random_seed)
    logreg.fit(X_normalized, y_label)
    if not os.path.exists('Trained_ML_models'):
        os.mkdir('Trained_ML_models')
    model_name = 'Trained_ML_models/'+bacteria_string+'_logistic_regression_model.pkl'
    with open(model_name, 'wb') as model_file:
        pickle.dump(logreg, model_file)

      ############# Plotting ##############
    h = .02
    x_min, x_max = X_normalized[:, 0].min() - 1, X_normalized[:, 0].max() + 1
    y_min, y_max = X_normalized[:, 1].min() - 1, X_normalized[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    ##### Training data Prediction ############

    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    class_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}

    y_label_color= label_encoder.transform(data['bacteria']) 

    colors = ['orange', 'green', '#1f77b4', 'purple', '#ffcc00']
    additional_colors = [ '#4682b4', '#d2b48c', '#8a2be2', '#fa8072', '#6a5acd', '#20b2aa', '#778899', '#b0c4de', '#ffff54']
    unique_labels_length = len(np.unique(y_label))
    colors_needed = unique_labels_length - len(colors)
    if colors_needed > 0:
        colors.extend(additional_colors[:colors_needed])

    light_colors = [lighten_color(c) for c in colors]
    print("light colors are: ",light_colors)
    light_colors = ['#' + ''.join('%02x' % int(c * 255) for c in color) for color in light_colors]
    cmap_background=ListedColormap(light_colors)
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.8)
    region_centers = {}
    for class_index in np.unique(Z):
        indices = np.where(Z == class_index)
        x_center = np.mean(xx[indices])
        y_center = np.mean(yy[indices])
        region_centers[class_index] = (x_center, y_center)
    for class_index, center in region_centers.items():
        class_name = label_encoder.classes_[class_index]
        plt.text(center[0] + 0.01, center[1] - 0.01, class_name, fontsize=10, 
                ha='center', va='center', color='black')
    y_label= label_encoder.transform(data['bacteria']) 
    print(np.unique(y_label))
    class_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    # Normalizing the original data
    original_data_normalized = scaler.fit_transform(data[['mag', 'dir']])
    region_centers = {}
    
    # # Creating a scatter plot for normalized 'mag' vs 'dir' with different colors for different bacteria classes in the original data
    for index, color in zip(np.unique(y_label), colors):
        original_label = class_mapping[index]
        plt.scatter(original_data_normalized[y_label == index, 0],original_data_normalized[y_label == index, 1], c=colors[original_label],label=f'{original_label}')
    plt.legend()
    plt.xlabel('Mag')
    plt.ylabel('Dir')   
    f_name="Plots/ML_Plots/"+bacteria_string+"_train_ML.svg"
    img_bytes_io = io.BytesIO()
    plt.savefig(img_bytes_io, format='png')
    img_bytes_io.seek(0)
    img_base64 = base64.b64encode(img_bytes_io.read()).decode('utf-8')
    plt.savefig(f_name,format = 'svg')

    return img_base64


def lighten_color(hex_color, amount=0.5):
    """Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple."""
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[hex_color]
    except:
        c = hex_color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    #plt.savefig('C:/Users/rmg00/Downloads/epsfolder/original_combinations_opticsdata.eps',format = 'eps')

def testing(data):
    df=data
    data['mag'] = np.sqrt(data['Ypos']**2 + data['Xpos']**2) / data['Pow']
    data['dir'] = np.arctan2(data['Ypos'], data['Xpos'])
    unq_bact = data['bacteria'].unique()


######## Get the Model, Scaler, Label Encoder ######


    model_files = model_picker(unq_bact)
    model_location = "Trained_ML_models/"+model_files[0]
    scaler_files = scaler_picker(unq_bact)
    scaler_location = "StandardScalers/"+scaler_files[0]
    LE_files = LE_picker(unq_bact)
    LE_location = "LabelEncoders/"+LE_files[0]

    print(model_location,scaler_location,LE_location)

    with open(model_location, 'rb') as file:
        model = pickle.load(file)
    with open(scaler_location, 'rb') as file:
        scaler = pickle.load(file)
    new_data_normalized = scaler.transform(data[['mag', 'dir']])
    with open(LE_location, 'rb') as file:
        labelenc = pickle.load(file)

    predictions= model.predict(new_data_normalized)
    y_true = labelenc.fit_transform(data['bacteria'])
    accuracy = accuracy_score(y_true, predictions)
    conf_matrix = confusion_matrix(y_true, predictions)
    print("Accuracy is : ",accuracy)

def model_picker(unq_bact):
    pickle_files = [f for f in os.listdir('Trained_ML_models') if f.endswith('.pkl')]
    matching_files = []
    # Loop through each file
    for file_name in pickle_files:
        # Split the file name by '_'
        parts = file_name.split('_')
        # Check if any part of the file name is in your list X
        if any(part in unq_bact for part in parts):
            # If so, add to the list of matching files
            matching_files.append(file_name)
    return matching_files

def scaler_picker(unq_bact):
    pickle_files = [f for f in os.listdir('StandardScalers') if f.endswith('.pkl')]
    matching_files = []
    # Loop through each file
    for file_name in pickle_files:
        # Split the file name by '_'
        parts = file_name.split('_')
        # Check if any part of the file name is in your list X
        if any(part in unq_bact for part in parts):
            # If so, add to the list of matching files
            matching_files.append(file_name)
    return matching_files

def LE_picker(unq_bact):
    pickle_files = [f for f in os.listdir('LabelEncoders') if f.endswith('.pkl')]
    matching_files = []
    # Loop through each file
    for file_name in pickle_files:
        # Split the file name by '_'
        parts = file_name.split('_')
        # Check if any part of the file name is in your list X
        if any(part in unq_bact for part in parts):
            # If so, add to the list of matching files
            matching_files.append(file_name)
    return matching_files

    

