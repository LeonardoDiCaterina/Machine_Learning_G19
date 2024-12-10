import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA

class PreProcessor():
    """
    This class is used to preprocess the data before feeding it to the model 
    

    """
    def __init__(self, scale=True,fillna = True, method_to_fillna='mean'):
        
        self.status = 'init'
        
        self.scale = scale
        self.fillna = fillna
        self.path = '../Data'
        self.scaler = RobustScaler()
        self.encoder = OneHotEncoder()
        self.columns_to_encode = ['Gender','Attorney/Representative','Carrier Type Code','First Hearing held']
        self.columns_to_freq_encode = []#['County of Injury','Carrier Code_','Carrier Type Code','Country of Injury Code','District Code','Medical Fee Code','Nature of Injury Code','Part of Body Code']
        
        self.scaled_features =[]
        self.columns_to_scale = []
        self.columns_NOT_to_scale = []
        self.columns_to_drop = []
        self.columns_dropped = []
        self.col_to_fillna = []
        self.method_to_fillna = method_to_fillna
        self.start_features = []
        self.end_features = []
        
        self.pca = PCA()
        
        self.cat_features = []
        self.num_features = []
        self.date_features = []
        self.dummy_features = []
        self.fe_features = []
        self.code_features = []
        
        
        self.version = "7.3 10 dec 11:17 coffe at cantine"
        
        
    def set_start_features(self, df):
        """ check the standard end features"""
        self.start_features = df.columns
        print(f"start features: {self.start_features}")
        return
    def set_end_features(self, df):
        """ check the standard end features"""
        self.end_features = df.columns
        print(f"end features: {self.end_features}")
        return
    
    def check_start_features(self, df):
        """ check if the features of df are the same as the end features"""
        current_features = df.columns
        if len(self.start_features) == 0:
            print("no start features")   
            return
        if set(self.start_features) != set(current_features):
            print("features are not the same")
            try:
                df= df[self.start_features]
                print(f"the features that have been discarded are {set(current_features) - set(self.start_features)}")
            except Exception as e:
                print("could not solve the issue")
                print(e)
        if set(self.start_features) == set(current_features):
            print("features are the same")
        return
    
    
    
    def check_end_features(self, df):
        """ check if the features of df are the same as the start features"""
        current_features = df.columns
        if len(self.end_features) == 0:
            print("no end features")   
            return
        if set(self.end_features) != set(current_features):
            print("features are not the same")
            try:
                df= df[self.end_features]
                print(f"the features that have been discarded are {set(current_features) - set(self.end_features)}")
            except Exception as e:
                print("could not solve the issue")
                print(e)
        for col in self.get_date_features():
            if col not in self.end_features:
                print(f"{col} not in end features")
                self.remove_date_features(col)
        for col in self.get_cat_features():
            if col not in self.end_features:
                print(f"{col} not in end features")
                self.remove_cat_features(col)
        for col in self.get_num_features():
            if col not in self.end_features:
                print(f"{col} not in end features")
                self.remove_num_features(col)
        for col in self.get_dummy_features():
            if col not in self.end_features:
                print(f"{col} not in end features")
                self.remove_dummy_features(col)
        for col in self.get_columns_to_freq_encode():
            if col not in self.end_features:
                print(f"{col} not in end features")
                self.remove_columns_to_freq_encode(col)
        
        return
    

    
    def choose_columns_to_fillna(self, df):
        self.col_to_fillna = list(df.select_dtypes(include=['int64','int32', 'float64']).loc[:, df.isna().sum() > 0].columns)
        return self.col_to_fillna
    
    def choose_columns_to_scale(self, df):
        cols = df.select_dtypes(include=['int64','int32', 'float64']).columns
        # Identify columns to scale by excluding columns that should not be scaled
        self.columns_to_scale = list(set(cols) - set(self.columns_NOT_to_scale) - set(self.code_features) )
        #print("**************** columns to scale ****************")
        self.append_columns_to_scale('County of Injury')
        #print(f"columns to scale: {self.columns_to_scale}")
        
    def choose_columns_to_freq_encode(self, df):
        
        code_cols = self.get_code_features()
        for col in code_cols:
            self.append_columns_to_freq_encode(col)
        
    
    def select_method_to_fillna(self, method_to_fillna):
        
        if method_to_fillna not in ['mean', 'median', 'mode','zero']:
            print("Invalid method")
            return
        self.method = method_to_fillna
        return
    
    def get_columns_to_fillna(self):
        return self.col_to_fillna
        
    def get_columns_to_freq_encode(self):
        return self.columns_to_freq_encode 
        
    def get_columns_to_encode(self):
        return self.columns_to_encode
    
    def get_columns_to_scale(self):
        return self.columns_to_scale
    
    def get_columns_NOT_to_scale(self):
        return self.columns_NOT_to_scale
    
    def get_columns_to_drop(self):
        return self.columns_to_drop
    
    def get_dummy_features(self):
        return self.dummy_features

    def get_cat_features(self):
        return self.cat_features
    
    def get_num_features(self):
        return self.num_features
    
    def get_start_features(self):
        return self.start_features
    
    def get_end_features(self):
        return self.end_features
    
    def get_date_features(self):
        return self.date_features
    
    def get_fe_features(self):
        return self.fe_features
    
    def get_code_features(self):
        return self.code_features

    def get_scaled_features(self):
        return self.scaled_features
    
    
    def append_columns_to_encode(self, col=None):
        if self.status == 'solid':
            if col in self.get_columns_to_encode():
                return
            else:
                print(f"status: {self.status}")
                print("cannot append columns to encode")
            return
        try:
            if col in self.get_columns_to_encode():
                print(f"{col} already in columns to encode")
            else:
                self.columns_to_encode.append(col)
        except Exception as e:
            print(f"Error in appending column to encode: {e}")
    
    def append_columns_to_scale(self, col=None):
        if self.status == 'solid':     
            if col not in self.scaler.get_feature_names_out():
                print(f"status: {self.status}")
                print(f"cannot {col} append columns to scale")
                return
        try:
            if col in self.get_columns_to_scale():
                print(f"{col} already in columns to scale")
            else:
                self.columns_to_scale.append(col)
        except Exception as e:
            print(f"Error in appending column to scale: {e}")
                    
                    
            
    def append_columns_NOT_to_scale(self, col=None):
        if self.status == 'solid':
            if col in self.get_columns_NOT_to_scale():
                return
            else:
                print(f"status: {self.status}")
                print("cannot append columns NOT to scale")
            return
        try:
            if col in self.get_columns_NOT_to_scale():
                print(f"{col} already in columns not to scale")
            else:
                self.columns_NOT_to_scale.append(col)
        except Exception as e:
            print(f"Error in appending column not to scale: {e}")
    
    def append_columns_to_drop(self, col=None):
        if self.status == 'solid' and col in self.get_end_features():
                print(f"status: {self.status}")
                print("cannot append columns to drop because they are in end features")
                return
        try:
            if col in self.get_columns_to_drop():
                print(f"{col} already in columns to drop")
            else:
                self.columns_to_drop.append(col)
        except Exception as e:
            print(f"Error in appending column to drop: {e}")
    
    def append_columns_to_freq_encode(self, col=None):
        if self.status == 'solid':
            if col in self.get_columns_to_freq_encode():
                return
            else:
                print(f"status: {self.status}")
                print("cannot append columns to frequency encode")
            return
        try:
            if col in self.get_columns_to_freq_encode():
                print(f"{col} already in columns to frequency encode")
            else:
                self.columns_to_freq_encode.append(col)
        except Exception as e:
            print(f"Error in appending column to frequency encode: {e}")
    
    def append_cat_features(self, col=None):
        if self.status == 'solid':
            if col in self.get_cat_features():
                return
            else:
                print(f"status: {self.status}")
                print("cannot append columns to cat features")
            return
        try:
            if col in self.get_cat_features():
                print(f"{col} already in cat features")
            else:
                self.cat_features.append(col)
        except Exception as e:
            print(f"Error in appending column to cat features: {e}")
            
    def append_num_features(self, col=None):
        if self.status == 'solid':
            if col in self.get_num_features():
                return
            else:
                print(f"status: {self.status}")
                print("cannot append columns to num features")
            return
        try:
            if col in self.get_num_features():
                print(f"{col} already in num features")
            else:
                self.num_features.append(col)
        except Exception as e:
            print(f"Error in appending column to num features: {e}")
            
    def append_date_features(self, col=None):
        if self.status == 'solid':
            if col in self.get_date_features():
                return
            else:
                print(f"status: {self.status}")
                print("cannot append columns to date features")
            return
        try:
            if col in self.get_date_features():
                print(f"{col} already in date features")
            else:
                self.date_features.append(col)
        except Exception as e:
            print(f"Error in appending column to date features: {e}")    
    
    def append_dummy_features(self, col=None):
        if self.status == 'solid':
            if col in self.get_dummy_features():
                return
            else:
                print(f"status: {self.status}")
                print("cannot append columns to dummy features")
            return
        try:
            if col in self.get_dummy_features():
                print(f"{col} already in dummy features")
            else:
                self.dummy_features.append(col)
        except Exception as e:
            print(f"Error in appending column to dummy features: {e}")
    
    def append_fe_features(self, col=None):
        if self.status == 'solid':
            if col in self.get_fe_features():
                return
            else:
                print(f"status: {self.status}")
                print("cannot append columns to fe features")
            return
        try:
            if col in self.get_fe_features():
                print(f"{col} already in fe features")
            else:
                self.fe_features.append(col)
        except Exception as e:
            print(f"Error in appending column to fe features: {e}")
            
    def append_scaled_feature(self, col=None):
        try:
            if col in self.get_scaled_features():
                print(f"{col} already in scaled features")
            else:
                self.scaled_features.append(col)
        except Exception as e:
            print(f"Error in appending column to scaled features: {e}")
            
    def append_code_features(self, col=None):
        if self.status == 'solid':
            if col in self.get_code_features():
                return
            else:
                print(f"status: {self.status}")
                print("cannot append columns to code features")
            return
        try:
            if col in self.get_code_features():
                print(f"{col} already in code features")
            else:
                self.code_features.append(col)
        except Exception as e:
            print(f"Error in appending column to code features: {e}")
            
    def remove_columns_to_encode(self, col=None): 
        try:
            if col in self.get_columns_to_encode():
                self.columns_to_encode.remove(col)
            else:
                print(f"{col} not in columns to encode")
        except Exception as e:
            print(f"Error in removing column to encode: {e}")
    
    def remove_columns_to_scale(self, col=None):
        try:
            if col in self.get_columns_to_scale():
                self.columns_to_scale.remove(col)
            else:
                print(f"{col} not in columns to scale")
        except Exception as e:
            print(f"Error in removing column to scale: {e}")
    
    def remove_columns_NOT_to_scale(self, col=None):
        try:
            if col in self.get_columns_NOT_to_scale():
                self.columns_NOT_to_scale.remove(col)
            else:
                print(f"{col} not in columns not to scale")
        except Exception as e:
            print(f"Error in removing column not to scale: {e}")
    
    def remove_columns_to_drop(self, col=None):
        try:
            if col in self.get_columns_to_drop():
                self.columns_to_drop.remove(col)
            else:
                print(f"{col} not in columns to drop")
        except Exception as e:
            print(f"Error in removing column to drop: {e}")
    
    def remove_columns_to_freq_encode(self, col=None):
        try:
            if col in self.get_columns_to_freq_encode():
                self.columns_to_freq_encode.remove(col)
            else:
                print(f"{col} not in columns to frequency encode")
        except Exception as e:
            print(f"Error in removing column to frequency encode: {e}")
            
    def remove_cat_features(self, col=None):
        try:
            if col in self.get_cat_features():
                self.cat_features.remove(col)
            else:
                print(f"{col} not in cat features")
        except Exception as e:
            print(f"Error in removing column to cat features: {e}")
    
    def remove_num_features(self, col=None):
        try:
            if col in self.get_num_features():
                self.num_features.remove(col)
            else:
                print(f"{col} not in num features")
        except Exception as e:
            print(f"Error in removing column to num features: {e}")
    
    def remove_date_features(self, col=None):
        try:
            if col in self.get_date_features():
                self.date_features.remove(col)
            else:
                print(f"{col} not in date features")
        except Exception as e:
            print(f"Error in removing column to date features: {e}")
    
    def remove_dummy_features(self, col=None):
        try:
            if col in self.get_dummy_features():
                self.dummy_features.remove(col)
            else:
                print(f"{col} not in dummy features")
        except Exception as e:
            print(f"Error in removing column to dummy features: {e}")    
    
    def remove_fe_features(self, col=None):
        try:
            if col in self.get_fe_features():
                self.fe_features.remove(col)
            else:
                print(f"{col} not in fe features")
        except Exception as e:
            print(f"Error in removing column to fe features: {e}")
            
    
    def remove_code_features(self, col=None):
        try:
            if col in self.get_code_features():
                self.code_features.remove(col)
            else:
                print(f"{col} not in code features")
        except Exception as e:
            print(f"Error in removing column to code features: {e}")
            
    
    def to_csv(self, df, file_name = 'cleaned_data'):
        import os
        if not file_name.endswith('.csv'):
            file_name = file_name + '.csv'
        return df.to_csv(os.path.join(self.path, file_name), index=False)
       
        
    
    """def freq_encode(self,df, col):
        encoding = df.groupby(col).size()
        # implement the duplicates later
        df[col] =  df[col].map(encoding)"""
    
    def freq_encode_features(self,df):
        self.choose_columns_to_freq_encode(df)
        cols = self.get_columns_to_freq_encode()
        #print(f"freq_encode_features ---> {cols}")
        for col in cols:
            fe_name = col + '_fe'
            try:
                df[fe_name] = df[col].map(df[col].value_counts())
            except:
                print(f"could not create {fe_name}")
            self.append_fe_features(fe_name)
            self.append_columns_NOT_to_scale(col)
        return df

        

    def __str__(self):
        return (
            #f"PreProcessor name: {self.__class__.__name__}\n"
            f"encoder: {self.encoder}\n"
            +f"scaler: {self.scaler}\n"
            +f"columns to encode: {self.get_columns_to_encode()}\n"
            +f"columns to scale: {self.get_columns_to_scale()}\n"
            +f"columns to drop: {self.get_columns_to_drop()}\n"
            +f"columns dropped: {self.columns_dropped}\n"
            +f"columns to fillna: {self.get_columns_to_fillna()}\n"
            +f"method to fillna: {self.method_to_fillna}\n"
            +f"file path for saving: {self.path}"
        )
    def cleanUp (self, df, to_csv = False, to_csv_name = 'cleaned_data.csv', to_csv_path = '../Data', fillna = False, dropna = False, pca = False, n_components = None):
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
        self.append_columns_to_drop('WCB Decision')
                
        # Numerical Features
        print("1)-------Numerical Features---------")
        ## Average Weekly Wage
        #print("A) ---> dealing with the Average Weekly Wage")
        df["log_Average Weekly Wage"] = np.log(df["Average Weekly Wage"]+1)
        self.append_columns_to_drop('Average Weekly Wage')
        self.append_num_features('log_Average Weekly Wage')

        ## Birth Year
        #print("B) ---> dealing with the Birth Year")
        nan_treshold_Birth_Year = 1900
        df["Birth Year"] = df["Birth Year"].apply(lambda x: x if x > nan_treshold_Birth_Year else np.nan)
        df["Age_"] = 2019 - df["Birth Year"]
        self.append_columns_to_drop('Birth Year')
        self.append_num_features('Age_')
        #print("created ---> Age_")

        ## IM4 Count
        df.fillna({'IME-4 Count': 0}, inplace=True)
        self.append_columns_NOT_to_scale('IME-4 Count')
        self.append_num_features('IME-4 Count')

        # Categorical Features
        print("2)-------Categorical Features---------")
        print("A) ---> dealing with the zip code")
        ## Zip Code -> freq encding would be nice
        df['Zip Code'] = df['Zip Code'].fillna('00000')
        df['Zip Code'] = df['Zip Code'].str[:5]
        #df['Zip Code'] = df['Zip Code'].apply(lambda x: int(x) if x.isnumeric() else 0)
        #print("created ---> Zip Code")
        self.append_code_features('Zip Code')
        
        
        cat_col = df.select_dtypes(include=['object']).columns
        


        #print("B) ---> dealing with Alternative Dispute Resolution")
        try:
            df['Alternative Dispute Resolution'] = df['Alternative Dispute Resolution'].map({'Y': 1, 'N': 0})
            self.append_columns_NOT_to_scale('Alternative Dispute Resolution')  
        except Exception as e:
            print(e)

        # encodng the categorical features
        #print("C) ---> encoding the categorical features")

        ## import the lookup tables
        #print("D) --->importing the lookup tables:")
        lookup_Carrier = pd.read_csv('../Data/lookup_carrier.csv')
        #print("1. ---> lookup_carrier")

        lookup_Carrier_Type = pd.read_csv('../Data/lookup_carrier_type.csv')
        #print("2. ---> lookup_carrier_type")

        lookup_Cause_of_Injury = pd.read_csv('../Data/lookup_cause_of_injury.csv')
        #print("3. ---> lookup_cause_of_injury")

        lookup_Country = pd.read_csv('../Data/lookup_country.csv')
        #print("4. ---> lookup_country")

        lookup_District = pd.read_csv('../Data/lookup_district.csv')
        #print("5. ---> lookup_district")

        lookup_Industry_Code = pd.read_csv('../Data/lookup_industry_code.csv')
        #print("6. ---> lookup_industry_code")

        lookup_Medical_fee = pd.read_csv('../Data/lookup_medical_fee.csv')
        #print("7. ---> lookup_medical_fee")

        lookup_Nature_of_Injury = pd.read_csv('../Data/lookup_nature_of_injury.csv')
        #print("8. ---> lookup_nature_of_injury")

        lookup_Part_of_Body = pd.read_csv('../Data/lookup_part_of_body.csv')
        #print("9. ---> lookup_part_of_body")

        #print("4)-------encoding the categorical features: ---------")
        ## Carrier Code_
        try:
            df['Carrier Code_'] = df['Carrier Name'].map(lookup_Carrier.set_index('Carrier Name')['Carrier Code'])
            #print("1. ---> Carrier Name encoded in Carrier Code_")
            self.append_columns_to_drop('Carrier Name')
            self.append_columns_NOT_to_scale('Carrier Code_')
            self.append_code_features('Carrier Code_')
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
            df['Country of Injury Code_'] = df['Country of Injury'].map(lookup_Country.set_index('County Name')['County Code'])
            #print("3. ---> Country of Injury encoded in Country Code_")
            self.append_columns_to_drop('Country of Injury')
            self.append_columns_NOT_to_scale('Country of Injury')
            self.append_code_features('Country of Injury Code_')
        except Exception as e:
            print(e)

        ## covid 19
        try:
            df['COVID-19 Indicator'] = df['COVID-19 Indicator'].map({'Y':1, 'N':0})
            #print("4. ---> COVID-19 Indicator encoded")
            self.append_columns_NOT_to_scale('COVID-19 Indicator')
            self.append_dummy_features('COVID-19 Indicator')
        except Exception as e:
            print(e)
        
        ## Agreement Reached
        try:
            self.append_columns_NOT_to_scale('Agreement Reached')
            self.append_dummy_features('Agreement Reached')
        except Exception as e:
            print(e)

        ## District Code_
        try:
            df['District Code_'] = df['District Name'].map(lookup_District.set_index('District Name')['District Code_'])
            #print("5. ---> District Name encoded in District Code_")
            self.append_columns_NOT_to_scale('District Code_')
            self.append_columns_to_drop('District Name')
            self.append_code_features('District Code_')
            
        except Exception as e:
            print(e)

        ## Medical Fee Code_
        try:
            df['Medical Fee Code_'] = df['Medical Fee Region'].map(lookup_Medical_fee.set_index('Medical Fee Region')['Medical Fee Code'])
            #print("6. ---> Medical Fee Region encoded in Medical Fee Code_")
            self.append_columns_NOT_to_scale('Medical Fee Code_')
            self.append_columns_to_drop('Medical Fee Region')
            self.append_code_features('Medical Fee Code_')
        except Exception as e:
            print(e)   

        ## Nature of Injury Code_

        ## Carrier Type Code_
        try:
            df['Carrier Type Code'] = df['Carrier Type'].str[:2]
            #print("7. ---> Carrier Type encoded in Carrier Type Code_")
            self.append_columns_NOT_to_scale('Carrier Type Code')
            self.append_columns_to_drop('Carrier Type')
            self.append_code_features('Carrier Type Code')
            self.append_cat_features('Carrier Type Code')
            self.append_columns_to_freq_encode('Carrier Type Code')
        except Exception as e:
            print(e)
            
        ### WCIO Part of Body Code
        try:
            df['Part of Body Code_'] = df['WCIO Part of Body Code'].astype(str)
            self.append_cat_features('WCIO Part of Body Code')
            self.append_columns_NOT_to_scale('WCIO Part of Body Code')
            self.append_code_features('Part of Body Code_')
        except Exception as e:
            print(e)

        ## WCIO Nature of Injury Code
        try:
            df['Nature of Injury Code_'] = df['WCIO Nature of Injury Code'].astype(str)
            self.append_cat_features('WCIO Nature of Injury Code')
            self.append_columns_NOT_to_scale('WCIO Nature of Injury Code')
            self.append_code_features('Nature of Injury Code_')
        except Exception as e:
            print(e)
        
        ## WCIO Cause of Injury Code
        try:
            df['Cause of Injury Code_'] = df['WCIO Cause of Injury Code'].astype(str)
            self.append_cat_features('WCIO Cause of Injury Code')
            self.append_columns_NOT_to_scale('WCIO Cause of Injury Code')
            self.append_code_features('Cause of Injury Code_')
        except Exception as e:
            print(e)
            
        ## WCIO Industry Code
        try:
            df['Industry Code_'] = df['WCIO Industry Code'].astype(str)
            self.append_cat_features('WCIO Industry Code')
            self.append_columns_NOT_to_scale('WCIO Industry Code')
            self.append_code_features('Industry Code_')
        except Exception as e:
            print(e)
        
        ## Carrier Type Code_
        try:
            df['Carrier Type Code_'] = df['Carrier Type'].astype(str)
            #print("8. ---> Carrier Type encoded in Carrier Type Code_")
            self.append_columns_NOT_to_scale('Carrier Type Code_')
            self.append_columns_to_drop('Carrier Type')
            self.append_code_features('Carrier Type Code_')
        except Exception as e:
            print(e)


        # Date Features
        #print("5) -------Date Features---------")
        date_col = cat_col[cat_col.str.contains('Date')]
        #print("setting every date to date time")
        date_col = cat_col[cat_col.str.contains('Date')]
        for col in date_col:
            try :
                df[col] = pd.to_datetime(df[col], errors='coerce', format='%Y-%m-%d')
            except Exception as e:
                print(e)

        # New Date Features
        #print("6) -------New Date Features---------")
        #print("A) ---> Accident Date year and month")
        df['Accident Year_'] = df['Accident Date'].dt.year
        df['Accident Month_'] = df['Accident Date'].dt.month
        self.append_columns_to_drop('Accident Date')
        self.append_date_features('Accident Year_')
        self.append_date_features('Accident Month_')
        

        #print("B) ---> Assembly Date year and month")
        df['Assembly Year_'] = df['Assembly Date'].dt.year
        df['Assembly Month_'] = df['Assembly Date'].dt.month
        self.append_columns_to_drop('Assembly Date')
        self.append_date_features('Assembly Year_')
        self.append_date_features('Assembly Month_')

        #print("C) ---> C-2 year and month")
        df['C-2 Year_'] = df['C-2 Date'].dt.year
        df['C-2 Month_'] = df['C-2 Date'].dt.month
        self.append_columns_to_drop('C-2 Date')
        self.append_date_features('C-2 Year_')
        self.append_date_features('C-2 Month_')

        #print("D) ---> C-3 year and month")
        df['C-3 Year_'] = df['C-3 Date'].dt.year
        df['C-3 Month_'] = df['C-3 Date'].dt.month
        self.append_columns_to_drop( 'C-3 Date')
        self.append_date_features('C-3 Year_')
        self.append_date_features('C-3 Month_')

        #print("E) --->First Hearing year and month")
        df['First Hearing Year_'] = df['First Hearing Date'].dt.year
        df['First Hearing Month_'] = df['First Hearing Date'].dt.month
        df['First Hearing held'] = ~df['First Hearing Date'].isna()
        self.append_columns_to_drop('First Hearing Date')
        self.append_date_features('First Hearing Year_')
        self.append_date_features('First Hearing Month_')

        #print("7) -------appending Description columns to col_to_drop---------")
        desc_col = list(df.columns[df.columns.str.contains('Description')])
        for col in desc_col:
            #print(f"appending {col} type: {df[col].dtype}")
            self.append_columns_to_drop(col)
        #print(f"->{self.get_columns_to_drop()}")
        #print("8)-------Encoding the categorical features---------")
        # encoding Attorney/Representative	
        #print("A) ---> encoding Attorney/Representative")
       
        #df['Attorney/Representative'] = df['Attorney/Representative'].map({'Y':1, 'N':0})

        # encoding Gender with one hot encoding
        #print("B) ---> encoding Gender")
        #columns_to_encode = ['Gender','Attorney/Representative','Carrier Type Code','First Hearing held']
        #import one hot encoder
        #create the encoder
        """self.encoder.fit(df[self.get_columns_to_encode()])
        tr = self.encoder.transform(df[self.get_columns_to_encode()])
        print("shape of the transformed data: ", tr.shape)
        print("shape of the original data: ", self.encoder.get_feature_names_out(self.get_columns_to_encode()).shape)
        encoded_df = pd.DataFrame(tr)
        #print(encoded_df.head())
        print(encoded_df.shape)
        encoded_df = pd.DataFrame(tr, columns=self.encoder.get_feature_names_out(self.get_columns_to_encode()))
        df = pd.concat([df, encoded_df], axis=1)
        df.drop(columns=self.get_columns_to_encode(), inplace=True)
        print("----> encoding done")
        print(f"----- dropping the columns: {self.get_columns_to_encode()}")
        print(f"created the one hot encoded coluns: {encoded_df.columns}")"""

        #frequency encoding
        print("-------Frequency Encoding---------")
        print("A) ---> County of Injury")
        self.append_columns_to_freq_encode('County of Injury')
        self.append_columns_NOT_to_scale('County of Injury')
        self.append_cat_features ('County of Injury')
        
        return df

    def scaler_fit(self, df):
        if len(self.get_columns_to_scale()) == 0:
            self.choose_columns_to_scale(df)
        try:
            self.scaler = self.scaler.fit(df[self.get_columns_to_scale()])
            #df.loc[:, self.get_columns_to_scale()] = self.scaler.fit_transform(df[self.get_columns_to_scale()])
            print("scaler_fit----> scaling done")
        except Exception as e:
            print(e)    
        
        return df
    
    def pca_fit(self, df, n_components):
        try:
            self.pca = PCA(n_components=n_components)
            self.pca.fit(df)
            df = self.pca.transform(df)
        except Exception as e:
            print(e)
        print("----> PCA done")
        return df
    
    def use_scaler(self, df):
        try:
            df.loc[:, self.get_columns_to_scale()] = self.scaler.transform(df[self.get_columns_to_scale()])
        except Exception as e:
            print(e)
            return print('could not scale')
        print("----> scaling done")
        return df
    
    def drop_columns(self, df):
        list_of_columns = self.get_columns_to_drop().copy()
        for col in list_of_columns:
            try:
                print(f"dropping {col} : {df[col].dtype}")
                df.drop(columns=col, inplace=True)
                self.columns_dropped.append(col)
                self.columns_to_drop.remove(col)
            except Exception as e:
                print(f"could not drop {col}")
                print(e)
            
            print("----> columns dropped")
        print(f"columns not dropped: {self.get_columns_to_drop()}")
        return df
    
    def fit_pca(self, df, n_components):
        self.pca = PCA(n_components=n_components)
        self.pca.fit(df)
        return
    
    def use_pca(self, df):
        try:
            df = self.pca.transform(df)
        except Exception as e:
            print(e)
        print("----> PCA done")
        return df
    
    def pca_report(self):
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        print(f"cumsum: {cumsum}")
        plt.plot(cumsum)
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        plt.show()
    
    def encode_df(self, df):
        
        encoder = OneHotEncoder(sparse_output=False)
        encoder = encoder.fit(df[self.get_columns_to_encode()])
        self.encoder = self.encoder.fit(df[self.get_columns_to_encode()])
        encoded_col_df = pd.DataFrame(encoder.transform(df[self.get_columns_to_encode()]), columns=encoder.get_feature_names_out(self.get_columns_to_encode()))
        for col in encoder.get_feature_names_out(self.get_columns_to_encode()):
            self.append_columns_NOT_to_scale(col)
            self.append_dummy_features(col)
        print(f"----> encoding done! created: {encoded_col_df.columns}")
        for col in self.get_columns_to_encode():
            self.append_columns_to_drop(col)
            print(f"appending {col} unique values: {df[col].unique()}")
        encoded_col_df.index = df.index
        for col in encoded_col_df.columns:
            df[col] = encoded_col_df[col]
            self.append_columns_NOT_to_scale(col)
        #encoded_df = pd.concat([df, encoded_col_df], axis=1)

        return df
    
    def fillnans(self, df) -> pd.DataFrame:
        if len(self.get_columns_to_fillna()) == 0:
            print("No columns to fillna")
            return df
        #print(f"columns to fillna: {self.get_columns_to_fillna()}")
        for col in self.col_to_fillna:
            if self.method_to_fillna == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif self.method_to_fillna == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif self.method_to_fillna == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0])
            elif self.method_to_fillna == 'zero' or self.method_to_fillna == 0:
                df[col] = df[col].fillna(0)
            else:
                print("Invalid method")
        return df
    
    def pipeline(self, df, fit_scaler = True, pca = False, pca_fit = False, n_components = None, set_end_features = False): 
        if set_end_features == False:
            self.status = "solid"
        if set_end_features == True:
            self.status = "draft"
        self.check_start_features(df)
        df = self.cleanUp(df)
        print("-------------cleanUp-------------------")
        print(f"--------------{len(df)}----------------")
        print(f"nan count: {df.isna().sum().sum()}")
        print("--------------------------------")
        df = self.encode_df(df)
        print("------------encode_df--------------------")
        print(f"--------------{len(df)}----------------")
        print(f"nan count: {df.isna().sum().sum()}")
        print("--------------------------------")
        


        print("------------freq_encode_features--------------------")
        df = self.freq_encode_features(df)
        print(f"nan count: {df.isna().sum().sum()}")

        df = self.drop_columns(df)
        print("------------drop_columns--------------------")
        print(f"--------------{len(df)}----------------")
        print(f"nan count: {df.isna().sum().sum()}")
        print("--------------------------------")
        
        if self.scale == True:
            self.choose_columns_to_scale(df)
            col2s = self.get_columns_to_scale()
            for col in self.get_columns_to_scale():
                if df[col].dtype == 'object':
                    self.remove_columns_to_scale(col)
                    print(f"removing {col} from columns to scale because it is an object")
                if df[col].nunique() == 1:
                    self.remove_columns_to_scale(col)
                    print(f"removing {col} from columns to scale because it has only one unique value")
                if df[col].nunique() == 0:
                    self.remove_columns_to_scale(col)
                    print(f"removing {col} from columns to scale because it has no unique value")
                if col in self.get_code_features():
                    self.remove_columns_to_scale(col)
                    print(f"removing {col} from columns to scale because it is a code feature")
                
                if col in self.get_dummy_features():
                    self.remove_columns_to_scale(col)
                    print(f"removing {col} from columns to scale because it is a dummy feature")
                    
            if self.status == 'draft':
                for col in self.get_columns_to_scale():
                    self.append_scaled_feature(col)

            
            if self.status == 'solid':
                if  self.get_columns_to_scale() != self.get_scaled_features():
                    print("columns to scale and columns scaled are not the same")
                    print(f"the following columns are not scaled: {set(self.get_columns_to_scale()) - set(self.get_scaled_features())}")

            print("------------choose_columns_to_scale--------------------")
            for col in col2s:
                print(f"columns to scale: {col} type: {df[col].dtype}")
            cols = self.get_columns_to_scale()
        
            if fit_scaler == True:
                self.scaler_fit(df[cols])
                print("------------scaler_fit--------------------")
                print(f"nan count: {df.isna().sum().sum()}")
                print(f"--------------{len(df)}----------------")
                print("--------------------------------")
            df[cols] = self.use_scaler(df[cols])
            print("------------use_scaler--------------------")
            print(f"--------------{len(df)}----------------")
            print(f"nan count: {df.isna().sum().sum()}")
            print("--------------------------------")


        if set_end_features == True:
            self.set_end_features(df)
        if set_end_features == False:
            self.check_end_features(df)
        
        
        
        if self.fillna == True:
            self.choose_columns_to_fillna(df)
            cols = self.get_columns_to_fillna()
            df = self.fillnans(df)
            
        if (pca == True) and (n_components != None):
            if pca_fit == True:
                df = self.pca_fit(df, n_components)
            
            df = self.use_pca(df)
            print("------------use_pca--------------------")
            print(f"--------------{len(df)}----------------")
            print("--------------------------------")
            
        return df
    
    def __str__(self):
        return (
            #f"preProcesser name: {self.__class__.__name__}\n"
            f"version: {self.version}\n"
            +f"encoder: {self.encoder}\n"
            +f"scaler: {self.scaler}\n"
            +f"start features: {self.get_start_features()}\n"
            +f"end features: {self.get_end_features()}\n"
            +f"cat features: {self.get_cat_features()}\n"
            +f"num features: {self.get_num_features()}\n"
            +f"date features: {self.get_date_features()}\n"
            +f"dummy features: {self.get_dummy_features()}\n"
            +f"fe features: {self.get_fe_features()}\n"
            +f"columns to frequency encode: {self.get_columns_to_freq_encode()}\n"
            +f"columns to encode: {self.get_columns_to_encode()}\n"
            +f"columns to scale: {self.get_columns_to_scale()}\n"
            +f"columns to drop: {self.get_columns_to_drop()}\n"
            +f"columns dropped: {self.columns_dropped}\n"
            +f"columns to fillna: {self.get_columns_to_fillna()}\n"
            +f"method to fillna: {self.method_to_fillna}\n"
            +f"file path for saving: {self.path}"
        )
