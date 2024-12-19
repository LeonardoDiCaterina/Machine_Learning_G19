import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import log
from scipy.stats import chi2_contingency



def entropy(feature):
    '''
    Calculate the entropy of a feature.
    
    Parameters:
        feature (pd.Series): The feature to calculate the entropy.
        
    Returns:
        float: The entropy of the feature.
    '''
    return -np.sum([p*np.log2(p) for p in feature.value_counts(normalize=True)])


def two_col_entropy_corr(f1, f2, n_iter=30, n_frac=0.5):
    '''
    Calculate the correlation between the entropy of two columns.
    
    Parameters:
    
        f1 (pd.Series): The first feature.
        f2 (pd.Series): The second feature.
        n_iter (int): The number of iterations to calculate the correlation (default 30).
        n_frac (float): The fraction of the dataset to sample (default 0.5).
        
        Returns:
            float: The correlation between the entropy of the two columns.
            
        '''
    
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
    '''
    Calculate the correlation between the entropy of the columns of a dataframe.
    
    Parameters:
        df (pd.DataFrame): The dataset.
        n_iter (int): The number of iterations to calculate the correlation (default 30).
        n_frac (float): The fraction of the dataset to sample (default 0.5).
    
    Returns:
        pd.DataFrame: A dataframe containing the correlation between the entropy of the columns'''

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



def chi2_feature_importance(f1, target, significance_level=0.05, log=False):
    """
    Evaluate if a feature is important to predict the target using the Chi-squared test.

    Parameters:
        data (pd.DataFrame): The dataset containing the feature and target.
        feature (str): The name of the feature column.
        target (str): The name of the target column.
        significance_level (float): The threshold for statistical significance (default 0.05).

    Returns:
        dict: A dictionary containing the chi-squared statistic, p-value, and whether the feature is important.
    """
    # Create a contingency table
    contingency_table = pd.crosstab(f1, target)

    # Perform the Chi-squared test
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

    total_observations = contingency_table.sum().sum()
    rows, cols = contingency_table.shape

    cramers_v = np.sqrt(
        chi2_stat / (total_observations * min(cols - 1, rows - 1)))

    is_important = p_value < significance_level

    if log:
        if cramers_v < 0.3:
            print(f" {is_important} the test is not significant {cramers_v}")
        elif cramers_v < 0.5:
            print(f" {is_important} the test is weakly significant {cramers_v}")
        elif cramers_v < 0.7:
            print(f" {is_important} the test is moderately significant {cramers_v}")
        elif cramers_v < 1:
            print(f" {is_important} the test is highly significant {cramers_v}")
        else:
            print(f"the test has no sense {cramers_v}")

    # Determine if the feature is important
    is_important = p_value < significance_level
    return cramers_v, p_value


# Create a string
# Save the model
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def report_and_save (model,model_name,y_pred, y,feature_list = None, model_filepath = '../Models'
                     , report_filepath = '../Reports', print_report = True ): 
    '''
    Save the model  as a pickle file and create a report as a text file.
    
    Parameters:
        model (object): The model to save.
        model_name (str): The name of the model.
        y_pred (array): The predicted values.
        y (array): The true values.
        feature_list (list): The list of features.
        model_filepath (str): The path to save the model (default '../Models').
        report_filepath (str): The path to save the report (default '../Reports').
        print_report (bool): Whether to print the report (default
    
    Returns:
        None
        
    Print:
        The accuracy, precision, recall, and confusion matrix.
    
    Please note that the model must have a get_params method to get the model parameters.
    '''
    
    model_parameters = " "
    try:
        model_parameters = model.get_params()
    except Exception as e:
        print(f"An error occurred while getting the model parameters: {e}")
    
    model_filename = model_name + '.sav'
    full_model_filename_os = os.path.join(model_filepath, model_filename)
    report_filename = model_name + '.txt'
    full_report_filename_os = os.path.join(report_filepath, report_filename)
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    cm = confusion_matrix(y, y_pred, normalize='pred')

    try:
        with open(full_model_filename_os, 'wb') as file:
            pickle.dump([cm[1][1], cm[0][0]], file)
            pickle.dump(feature_list, file)
            pickle.dump(model, file)
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")

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
        
        
        

