import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import log

def entropy(feature):
    return -np.sum([p*np.log2(p) for p in feature.value_counts(normalize=True)])


def two_col_entropy_corr(f1, f2, n_iter=30, n_frac=0.5):
    baseline_s_x1 = entropy(f1)
    baseline_s_x2 = entropy(f2)

    s_x1 = []
    s_x2 = []
    for i in range(n_iter):
        x1 = f1.sample(frac=0.9, random_state=i)
        x2 = f2.sample(frac=0.9, random_state=i)
        s_x1.append(baseline_s_x1 - entropy(x1)/baseline_s_x1)
        s_x2.append(baseline_s_x2 - entropy(x2)/baseline_s_x2)
    return np.corrcoef(s_x1, s_x2)[0, 1]



def entropy_corr(df, n_iter=30, n_frac=0.5):

    entropy_matrix = []
    for col in df.columns:
        s = entropy(df[col])
        delta_entropy = []
        for i in range(n_iter):
            x = df[col].sample(frac=n_frac, random_state=i)
            delta_entropy.append(s - entropy(x) / s)
        entropy_matrix.append(delta_entropy)
    entropy_matrix = pd.DataFrame(entropy_matrix, index=df.columns)
    entropy_matrix = entropy_matrix.T.corr()
    return entropy_matrix.applymap(lambda x: log((x + 1) / 2))


# Create a string
# Save the model
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def report_and_save (model,model_name,y_pred, y,feature_list = None, model_filepath = '../Models'
                     , report_filepath = '../Reports', print_report = True ): 
    
    
    model_parameters = " "
    try:
        model_parameters = model.get_params()
    except Exception as e:
        print(f"An error occurred while getting the model parameters: {e}")
    
    model_filename = model_name + '.sav'
    full_model_filename_os = os.path.join(model_filepath, model_filename)
    report_filename = model_name + '.txt'
    full_report_filename_os = os.path.join(report_filepath, report_filename)

    try:
        with open(full_model_filename_os, 'wb') as file:
            pickle.dump(feature_list, file)
            pickle.dump(model, file)
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    cm = confusion_matrix(y, y_pred, normalize='true')

    # Save the string into a text file
    try:
        with open(full_report_filename_os, "w") as file:
            file.write(model_name)
            file.write("\n________________________\n")
            file.write('model_parameters: \n')
            file.write(str(model_parameters).replace(",", "\n"))   
            file.write("\n________________________\n")
            file.write('Accuracy: ')
            file.write(str(accuracy))
            file.write("\n________________________\n")
            file.write('Precision: ')
            file.write(str(precision))
            file.write("\n________________________\n")
            file.write('Recall: ')
            file.write(str(recall))
            file.write("\n________________________\n")
            file.write('Confusion Matrix:\n')
            file.write(str(cm))
            file.write("\n________________________\n")
    except Exception as e:
        print(f"An error occurred while writing the report: {e}")
    
    if print_report:
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, cmap='coolwarm')
        plt.title(f'Confusion Matrix {model_name}')   
        
        
        

