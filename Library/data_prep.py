import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# frequency encoding
def freq_encode(df, col):
    encoding = df.groupby(col).size()
    df[col] = df[col].map(encoding)
    # implement the duplicates later
    return df

def cleanUp (df, to_csv = False, to_csv_name = 'cleaned_data.csv', to_csv_path = '../Data', scale = False, scaler_used = None, fillna = False, dropna = False, pca = False, n_components = None):
    """
    This function cleans the data and prepares it for the model

    Parameters:
        df(DataFrame): The data to be cleaned
        to_csv(bool): If True, the cleaned data will be saved as a csv file
        to_csv_name(str): The name of the csv file
        to_csv_path(str): The path where the csv file will be saved
        scale(bool): If True, the data will be scaled
        fillna(bool): If True, the missing values will be filled
        dropna(bool): If True, the missing values will be dropped
        
    Returns:
        DataFrame: The cleaned data
    
    Example of usage:
        sys.path.append('../Library')
        import data_prep as p
        df = pd.read_csv('../Data/data.csv')
        df = p.cleanUp(df, to_csv = True)
        
       
        

    """    
    col_to_drop = []
    non_scalable_col = []
    # Numerical Features
    print("-------Numerical Features---------")
    num_col = df.select_dtypes(include=['int64', 'float64']).columns
    ## Average Weekly Wage
    print("A) ---> dealing with the Average Weekly Wage")
    df["log_Average Weekly Wage"] = np.log(df["Average Weekly Wage"]+1)
    col_to_drop.append('Average Weekly Wage')
    
    ## Birth Year
    print("B) ---> dealing with the Birth Year")
    nan_treshold_Birth_Year = 1900
    df["Birth Year"] = df["Birth Year"].apply(lambda x: x if x > nan_treshold_Birth_Year else np.nan)
    df["Age_"] = 2019 - df["Birth Year"]
    print("created ---> Age_")
    
    ## IM4 Count
    df.fillna({'IME-4 Count': 0}, inplace=True)
    
    # Categorical Features
    print("-------Categorical Features---------")
    cat_col = df.select_dtypes(include=['object']).columns
    ## drop the descriptions
    col_to_drop.extend(cat_col[cat_col.str.contains('Description')])
    
    print("B) ---> dealing with Alternative Dispute Resolution")
    try:
        df['Alternative Dispute Resolution'] = df['Alternative Dispute Resolution'].map({'Y': 1, 'N': 0})
    except Exception as e:
        print(e)
    
    # encodng the categorical features
    print("C) ---> encoding the categorical features")
    
    ## import the lookup tables
    print("importing the lookup tables:")
    lookup_Carrier = pd.read_csv('../Data/lookup_carrier.csv')
    print("1. ---> lookup_carrier")
    
    lookup_Carrier_Type = pd.read_csv('../Data/lookup_carrier_type.csv')
    print("2. ---> lookup_carrier_type")
    
    lookup_Cause_of_Injury = pd.read_csv('../Data/lookup_cause_of_injury.csv')
    print("3. ---> lookup_cause_of_injury")
    
    lookup_Country = pd.read_csv('../Data/lookup_country.csv')
    print("4. ---> lookup_country")
    
    lookup_District = pd.read_csv('../Data/lookup_district.csv')
    print("5. ---> lookup_district")
    
    lookup_Industry_Code = pd.read_csv('../Data/lookup_industry_code.csv')
    print("6. ---> lookup_industry_code")
    
    lookup_Medical_fee = pd.read_csv('../Data/lookup_medical_fee.csv')
    print("7. ---> lookup_medical_fee")
    
    lookup_Nature_of_Injury = pd.read_csv('../Data/lookup_nature_of_injury.csv')
    print("8. ---> lookup_nature_of_injury")
    
    lookup_Part_of_Body = pd.read_csv('../Data/lookup_part_of_body.csv')
    print("9. ---> lookup_part_of_body")
    
    print("encoding the categorical features:")
    ## Carrier Code_
    try:
        df['Carrier Code_'] = df['Carrier Name'].map(lookup_Carrier.set_index('Carrier Name')['Carrier Code'])
        print("1. ---> Carrier Name encoded in Carrier Code_")
        col_to_drop.append('Carrier Name')
    except Exception as e:
        print(e)
    
    ## Carrier Type Code_
    """"    try: 
        df['Carrier Type Code_'] = df['Carrier Type'].str[:2]
        print("2. ---> Carrier Type encoded in Carrier Type Code_")
        col_to_drop.append('Carrier Type')
    except Exception as e:
        print(e) 
    
    """
    print("2. ---> Carrier Type encoded in Carrier Type Code")
        
    ##County Code_
    try:
        df['Country of Injury'] = df['Country of Injury'].map(lookup_Country.set_index('County Name')['County Code'])
        print("3. ---> Country of Injury encoded in Country Code_")
        col_to_drop.append('Country of Injury')
    except Exception as e:
        print(e)
    
    ## covid 19
    try:
        df['COVID-19 Indicator'] = df['COVID-19 Indicator'].map({'Y':1, 'N':0})
        print("4. ---> COVID-19 Indicator encoded")
    except Exception as e:
        print(e)
    
    ## District Code_
    try:
        df['District Code'] = df['District Name'].map(lookup_District.set_index('District Name')['District Code'])
        print("5. ---> District Name encoded in District Code_")
        col_to_drop.append('District Name')
    except Exception as e:
        print(e)
    
    ## Medical Fee Code_
    try:
        df['Medical Fee Code_'] = df['Medical Fee Region'].map(lookup_Medical_fee.set_index('Medical Fee Region')['Medical Fee Code'])
        print("6. ---> Medical Fee Region encoded in Medical Fee Code_")
        col_to_drop.append('Medical Fee Region')
    except Exception as e:
        print(e)   
        
    ## Nature of Injury Code_
    
    ## Carrier Type Code_
    try:
        df['Carrier Type Code'] = df['Carrier Type'].str[:2]
        print("7. ---> Carrier Type encoded in Carrier Type Code_")
        col_to_drop.append('Carrier Type')
    except Exception as e:
        print(e)
    
        
    
    # Date Features
    print("-------Date Features---------")
    date_col = cat_col[cat_col.str.contains('Date')]
    print("setin every date to date time")
    date_col = cat_col[cat_col.str.contains('Date')]
    for col in date_col:
        try :
            df[col] = pd.to_datetime(df[col], errors='coerce', format='%Y-%m-%d')
        except Exception as e:
            print(e)
    
    # New Date Features
    print("-------New Date Features---------")
    print("A) ---> Accident Date year and month")
    df['Accident Year_'] = df['Accident Date'].dt.year
    df['Accident Month_'] = df['Accident Date'].dt.month
    col_to_drop.append('Accident Date')
    
    print("B) ---> Assembly Date year and month")
    df['Assembly Year_'] = df['Assembly Date'].dt.year
    df['Assembly Month_'] = df['Assembly Date'].dt.month
    col_to_drop.append('Assembly Date')
    
    print("C) ---> C-2 year and month")
    df['C-2 Year_'] = df['C-2 Date'].dt.year
    df['C-2 Month_'] = df['C-2 Date'].dt.month
    col_to_drop.append('C-2 Date')
    
    print("D) ---> C-3 year and month")
    df['C-3 Year_'] = df['C-3 Date'].dt.year
    df['C-3 Month_'] = df['C-3 Date'].dt.month
    col_to_drop.append('C-3 Date')
    
    print("E) --->First Hearing year and month")
    df['First Hearing Year_'] = df['First Hearing Date'].dt.year
    df['First Hearing Month_'] = df['First Hearing Date'].dt.month
    df['First Hearing held'] = ~df['First Hearing Date'].isna()
    col_to_drop.append('First Hearing Date')
    
    print("----> dropping the columns")
    df.drop(columns=col_to_drop, inplace=True)
    print("----> columns dropped")
    for col in df.columns:
        print(f"{col} : {df[col].dtype}")
    print("----> cleaning up done")
    
    #encoding the categorical features
    print("-------Encoding the categorical features---------")
    # encoding Attorney/Representative	
    print("A) ---> encoding Attorney/Representative")
    df['Attorney/Representative'] = df['Attorney/Representative'].map({'Y':1, 'N':0})
    
    # encoding Gender with one hot encoding
    print("B) ---> encoding Gender")
    columns_to_encode = ['Gender','Attorney/Representative','WCB Decision','Carrier Type Code','First Hearing held']
    #import one hot encoder
    #create the encoder
    encoder = OneHotEncoder(sparse_output=False)
    encoder = encoder.fit(df[columns_to_encode])
    encoded_df = pd.DataFrame(encoder.transform(df[columns_to_encode]), columns=encoder.get_feature_names_out(columns_to_encode))
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(columns=columns_to_encode, inplace=True)
    print("----> encoding done")
    print(f"----- dropping the columns: {columns_to_encode}")
    print(f"created the one hot encoded coluns: {encoded_df.columns}")
    
    #frequency encoding
    print("-------Frequency Encoding---------")
    print("A) ---> County of Injury")
    freq_encode(df, 'County of Injury')
    
    if scale and scaler_used:
        num_col = df.select_dtypes(include=['int64', 'float64']).columns
        print("-------Scaling the data---------")
        print("----> scaling the data")
        if scaler_used != None :
            df[num_col] = scaler_used[num_col].fit_transform(df[num_col])
        else:
            try:
                scaler_used = RobustScaler().fit(df[num_col])
                df[num_col] = scaler_used.fit_transform(df[num_col])
            except Exception as e:
                if not isinstance(scaler_used, RobustScaler):
                    print('Scaler is not a RobustScaler')
                print(e)    
        print("----> scaling done")
    if fillna:
        print("-------Filling the missing values---------")
        print("----> filling the missing values")
        df.fillna(method='ffill', inplace=True)
        print("----> filling done")
    if dropna:
        print("-------Dropping the missing values---------")
        print("----> dropping the missing values")
        df.dropna(inplace=True)
        print("----> dropping done")
    
    if pca:
        print("-------PCA---------")
        print("----> applying PCA")
        pca = PCA(n_components=n_components)
        df_pca = pca.fit_transform(df)
        
        plt.figure(figsize=(10,10))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)') #for each component
        plt.title('Explained Variance')
        plt.show()
        
        pca.components_
        
        print("----> PCA done")
        
            
    if to_csv:
        if scale:
            to_csv_name = 'scaled_' + to_csv_name
        path_name = to_csv_path + to_csv_name
        df.to_csv(path_name, index=False)
        print(f"----> saved the cleaned data in {path_name}")
    
    if scale:
            print("----> returning the scaler data")
            return df,scaler_used
    else:
        return df
    
    
    
         
    
    
    
                         
                        