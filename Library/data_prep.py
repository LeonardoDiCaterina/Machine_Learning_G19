#import robust scaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
#import one hot encoder
from sklearn.preprocessing import OneHotEncoder

class PreProcessor1:
    def __init__(self):
        
        self.status = 'train'
        self.scaler = RobustScaler()
        self.version = '4.1 lackOfFrango'
        self.encoder = OneHotEncoder()
        self.max_col_onehot = 5
        
        self.date_cols = []
        self.desc_cols = []
        self.code_cols = []
        self.casting_methods = []
        self.casting_list = [
    ('Age at Injury',                       'Int64'),
    ('Average Weekly Wage',                 'float64'),
    # ('WCB Decision',                        'remove'),
    ('Alternative Dispute Resolution',      'string-U-nan'),
    ('District Name',                       'none'),
    ('Carrier Name',                        'string'),
    ('IME-4 Count',                         'Int64'),
    ('COVID-19 Indicator',                  'string'),
    ('Number of Dependents',                'Int64'),
    ('Carrier Type',                         '.str[:2]'),
    ('Medical Fee Region',                   'string'),
    ('County of Injury',                    'string'),
    ('Agreement Reached',                   'Int64'),
    ('Attorney/Representative',             'string'),
    ('Birth Year',                          'Int64'),
    ('Gender',                              'string'),
    ('Zip Code',                            '.str[0:5]'),
]
        self.casted_cols = []
        self.transformation_list = [
    ('Age at Injury',                   'none',             '-'),
    ('Average Weekly Wage',             'log',              'log_Average Weekly Wage'),
    # ('WCB Decision',                    'none',             '-'),
    ('Alternative Dispute Resolution',  'dummy-YN',         'Alternative Dispute Resolution'),
    ('District Name',                   'none',             '-'),
    ('Carrier Name',                    'none',             '-'),
    ('IME-4 Count',                     'none',             '-'),
    ('COVID-19 Indicator',              'dummy-YN',         'COVID-19 Indicator'),
    ('Number of Dependents',            'none',             '-'),
    ('Carrier Type',                    'oneHot',           '-OneHot'),
    ('Carrier Type',                    'freq_encode',      'fe_Carrier Type'),
    ('Medical Fee Region',              'oneHot',           '-oneHot'),
    ('Medical Fee Region',              'freq_encode',      'fe_Medical Fee Region'),
    ('County of Injury',                'oneHot',           '-oneHot'),
    ('County of Injury',                'freq_encode',      'fe_County of Injury'),
    ('Agreement Reached',               'none',             '-oneHot'),
    ('Attorney/Representative',         'dummy-YN',         'Attorney/Representative'),
    ('Birth Year',                      'subtract_1900',    'Age'),
    ('Gender',                          'oneHot',           '-oneHot'),
]     
        self.transformed_cols = []
        self.fillna_list = []
        
        self.sclaing_list = [   ('Carrier Type',                     0),
                                ('Zip Code',                         0),
                                ('log_Average Weekly Wage',          1),
                                ('Alternative Dispute Resolution',   0),
                                ('COVID-19 Indicator',               0),
                                ('fe_Carrier Type',                  1),
                                ('fe_Medical Fee Region',            1),
                                ('fe_County of Injury',              1),
                                ('Attorney/Representative',          0),
                                ('Age',                              1),
                                ('Assembly Month',                   1),
                                ('Assembly Year',                    1),
                                ('C-3 Month',                        1),
                                ('C-3 Year',                         1),
                                ('Accident Month',                   1),
                                ('Accident Year',                    1),
                                ('C-2 Month',                        1),
                                ('C-2 Year',                         1),
                                ('First Hearing Month',              1),
                                ('First Hearing Year',               1),
                                ('fe_WCIO Part Of Body Code',        1),
                                ('fe_Industry Code',                 1),
                                ('fe_WCIO Nature of Injury Code',    1),
                                ('fe_Zip Code',                      1),
                                ('fe_WCIO Cause of Injury Code',     1),
                                ('WCIO Part Of Body Code',           1),
                                ('Industry Code',                    1),
                                ('WCIO Nature of Injury Code',       1),
                                ('Gender_U',                         0),
                                ('Gender_X',                         0),
                                ('IME-4 Count',                      1),
                                ('Age at Injury',                    1),
                                ('District Name',                    0),
                                ('Average Weekly Wage',              1),
                                ('Medical Fee Region',               0),
                                ('Number of Dependents',             1),
                                ('Carrier Name',                     0),
                                ('Gender_F',                         0),
                                ('Agreement Reached',                0),
                                ('Gender',                           0),
                                ('Birth Year',                       1),
                                ('WCB Decision',                     0),
                                ('Gender_M',                         0),
                                ('WCIO Cause of Injury Code',        1),
                                ('County of Injury',                 0)]
        
        self.freq = []
        
    def appeend_freq_list(self, freq, col):
        self.freq.append((freq, col))


# ------------------------ internal functions
    def update_status(self, status):
        if status in ['train', 'valid', 'test']:
            self.status = status
        else:  
            print('Unknown status')
    
    def __str__(self):
        return (f"PreProcessor1: {self.status}" +
            f"scaler: {self.scaler}" +
                f"version: {self.version}")
        
        
       
        
# ------------------------casting part
        
        
    def update_casted_cols(self):     
        self.casted_cols = [col for col, method in self.casting_list]

    def update_casting_methods(self):
        self.casting_methods = [method for col, method in self.casting_list]

    def set_castings(self,df):
        df_cols = df.columns
        columns_to_be_casted = set(df_cols) - set(self.casted_cols)
    
        date_cols = []
        date_cols.extend([x for x in df_cols if 'Date' in x])
        for col in date_cols:
            self.append_casting(col, 'string')
        columns_to_be_casted = columns_to_be_casted - set(date_cols)
    
        desc_cols = []
        desc_cols.extend([x for x in df_cols if 'Description' in x])
        for col in desc_cols:
            self.append_casting(col, 'remove')
        columns_to_be_casted = columns_to_be_casted - set(desc_cols)
    
        code_cols = []
        code_cols.extend([x for x in df_cols if 'Code' in x])
        for col in code_cols:
            self.append_casting(col, 'Int64')
        columns_to_be_casted = columns_to_be_casted - set(code_cols)
    
        if len(columns_to_be_casted) > 0:
            print ('Columns that are not casted:')
            for col in columns_to_be_casted:
                try:
                    print (f'-{col}: {df[col].dtype}')
                except Exception as e: 
                    print(e)
                    print (f'-{col}: not found')

    def look_for_feature_casting(self,feature):
        i = 0
        for col, method in self.casting_list:
            if col == feature:
                print(f'-Column {col} is casted as {method} at index {i}')
                return method
            i += 1
        return None

    def update_casting_list(self,feature,method):
        i = 0
        for col, method in self.casting_list:
            if col == feature:
                self.casting_list[i] = (feature, method)
                return
            i += 1
        #print(f'-Column {feature} not found in casting list. Adding it now')
        self.append_casting(feature, method)
    
    def append_casting(self,feature,method):
        self.casting_list.append((feature, method))
    
        
    def cast_pipeline(self,df):
        for col, method in self.casting_list:
            if method == 'Int64':
                df = self.cast_Int64(df, col)
            elif method == 'float64':
                df = self.cast_Float64(df, col)
            elif method == 'string':
                df = self.cast_string(df, col)
            elif method == 'string-U-nan':
                df = self.cast_string_U_nan(df, col)
            elif method == '.str[:2]':
                df = self.cast_string_2(df, col)
                self.append_scaling(col, 0)
            elif method == 'datetime64':
                df = self.cast_datetime64(df, col)
                self.append_scaling(col, 0)
            elif method == 'remove':
                df = self.cast_remove(df, col)
            elif method == '.str[0:5]':
                df = self.cast_string_5(df, col)
                self.append_scaling(col, 0)
                
            elif method == 'none':
                pass
            else:
                print(f'Unknown method {method} for column {col}')
        return df

    def cast_Int64(self, df, col):
        try:
            df[col] = df[col].astype('Int64')
        except Exception as e:
            print(e)
            print(f'-Column {col} not found')
        return df


    def cast_Float64(self, df, col):
        try:
            df[col] = df[col].astype('float64')
        except Exception as e:
            print(e)
            print(f'-Column {col} not found')

        return df

    def cast_string(self, df, col):
        try:
            df[col] = df[col].astype('string')
        except Exception as e:
            print(e)
            print(f'-Column {col} not found')

        return df

    def cast_string_U_nan(self, df, col):
        try:
            df[col] = df[col].astype('string')
            df[col] = df[col].replace('U', 'N')
        except Exception as e:
            print(e)
            print(f'-Column {col} not found')
        return df

    def cast_string_2(self, df, col):
        try:
            df[col] = df[col].astype('string')
            df[col] = df[col].str[:2]
        except Exception as e:
            print(e)
            print(f'-Column {col} not found')
        
        return df
    
    def cast_string_5(self, df, col):
        try:
            df[col] = df[col].astype('string')
            df[col] = df[col].fillna('00000')
            df[col] = df[col].str[:5]
            df[col] = df[col].apply(lambda x: int(x) if x.isnumeric() else 0)
            #df[col] = df[col].astype('string')
            
        except Exception as e:
            print(e)
            print(f'-Column {col} not found')
        
        return df


    def cast_datetime64(self, df, col):
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception as e:
            print(e)
            print(f'-Column {col} not found')
        
        return df


    def cast_remove(self, df, col):
        try:
            df.drop(columns=[col], inplace=True)
        except Exception as e:
            print(e)
            print(f'-Column {col} not found')
        return df
    
    
# ---------------------------------encoding part

    def refresh_transformed_cols(self):
        self.transformed_cols = [new_col for col, method, new_col in self.transformation_list]
        if len(self.transformed_cols) > len(set(self.transformed_cols)):
            print('Warning: there are duplicates in the transformation list')
    
    def update_transformation_methods(self):
        self.transformation_methods = set([method for col, method, new_col in self.transformation_list])
        
    def look_for_feature_transformation(self,feature):
        i = 0
        result = []
        for col, method, new_col in self.transformation_list:   
            if col == feature:
                print(f'-Column {col} is transformed in {new_col} trough {method} at index {i}')
                result.append((method, new_col, i))
            i += 1
        print(f'-Column {feature} not found in transformation list')
        return result
    
    def look_for_new_col(self,feature):
        i = 0
        for col, method, new_col in self.transformation_list:
            if col == feature:
                print(f'-Column {new_col} is made from {method} at index {i}')
                return new_col
            i += 1
        print(f'-Column {feature} not found in transformation list')
        return None
    
    def append_transformation(self,feature,method,new_col):
        self.transformation_list.append((feature, method, new_col))


    def transformation_pipeline(self,df):
        onehot_list = []
        for col, method, new_col in self.transformation_list:
            if method == 'log':
                df = self.transformation_log(df, col, new_col)
            elif method == 'subtract_1900':
                df = self.transformation_subtract_1900(df, col, new_col)
            elif method == 'dummy-YN':
                df = self.transformation_dummy_yn(df, col, new_col)
            elif method == 'freq_encode':
                df = self.transformation_freq_encode(df, col, new_col)
            elif method == 'oneHot':
                if len(df[col].unique()) < self.max_col_onehot:
                    onehot_list.append(col)
                else:
                    print(f'Column {col} has too many unique values for oneHot encoding: {len(df[col].unique())}')
                    print(f'ony {self.max_col_onehot} allowed, change the parameter if needed')
                #df = self.transformation_oneHot(df, col, new_col)
            elif method == '.str[5:7]':
                df = self.transformation_str57(df, col, new_col)
            elif method == '.str[0:4]':
                df = self.transformation_str04(df, col, new_col)
            elif method == '.str[0:5]':
                df = self.transformation_str05(df, col, new_col)
            elif method == 'remove':
                df = self.transformation_remove(df, col)
            elif method == 'none':
                pass
            else:
                print(f'Unknown method {method} for column {col}')
                
        if len(onehot_list) > 0:
            self.encoder.fit(df[onehot_list])
            X = self.encoder.transform(df[onehot_list])
            X = pd.DataFrame(X.toarray(), columns=self.encoder.get_feature_names_out(onehot_list), index = df.index)
            df = pd.concat([df, X], axis=1)
        return df     


    
    def set_transformations(self,df)->None:
        df_cols = df.columns
        self.refresh_transformed_cols()
        columns_to_be_transformed = set(df_cols) - set(self.transformed_cols)
    
        date_cols = []
        date_cols.extend([x for x in columns_to_be_transformed if 'Date' in x])
        for col in date_cols:
            col_year = col.replace('Date', 'Year')
            col_month = col.replace('Date', 'Month')
            self.append_transformation(col, '.str[5:7]', col_month)
            self.append_transformation(col, '.str[0:4]', col_year)
            self.append_transformation(col, 'remove', '')
        columns_to_be_transformed = columns_to_be_transformed - set(date_cols)
            
        code_cols = []
        code_cols.extend([x for x in columns_to_be_transformed if 'Code' in x])
        for col in code_cols:
            if len(df[col].unique()) < 10:
                self.append_transformation(col, 'oneHot', '-one')
            self.append_transformation(col, 'freq_encode', 'fe_'+col)
        
                    
            
        
        columns_to_be_transformed = columns_to_be_transformed - set(code_cols)
    
        if len(columns_to_be_transformed) > 0:
            print ('Columns that are not transformed:')
            for col in columns_to_be_transformed:
                try:
                    print (f'-{col}: {df[col].dtype}')
                except Exception as e: 
                    print(e)
                    print (f'-{col}: not found')
                    
                    
    def transformation_log(self, df, col, new_col):
        try:
            df[col].fillna(0, inplace=True)
            df[new_col] = np.log(df[col] + 1)
            self.append_scaling(new_col, 1)
        except Exception as e:
            print(e)
            print(f'Column {col} not found')
        return df


    def transformation_subtract_1900(self, df, col, new_col):
        try:
            df[new_col] = df[col] - 1900
            self.append_scaling(new_col, 1)
        except Exception as e:
            print(e)
            print(f'Column {col} not found')
        return df


    def transformation_dummy_yn(self, df, col, new_col):
        try:
            df[new_col] = df[col].apply(lambda x: 1 if x == 'Y' else 0)
            self.append_scaling(new_col, 0)    
        except Exception as e:
            print(e)
            print(f'Column {col} not found')
        return df


    def transformation_freq_encode(self, df, col, new_col):
        try:
            if self.status == 'train':
                freq = df[col].value_counts(normalize=True)
                # make a dictionary out of freq
                freq = freq.to_dict()
                self.freq.append((freq, col))
            else:
                for freq, col in self.freq:
                    if col == col:
                        freq = freq
            df.loc[:, new_col] = df[col].map(freq)
            # fillna with 0
            self.fillna_zero(df, new_col)
            
            self.append_scaling(new_col, 1)
        except Exception as e:
            print(e)
            print(f'Column {col} not found')
        return df


    def transformation_encode(self, df, col, new_col):
        try:
            df[new_col], _ = df[col].factorize()
        except Exception as e:
            print(e)
            print(f'Column {col} not found')
        return df


    def transformation_str57(self, df, col, new_col):
        try:
            df[new_col] = df[col].str[5:7].astype('Int64')
            self.append_scaling(new_col, 0)
        except Exception as e:
            print(e)
            print(f'Column {col} not found')
        
        return df


    def transformation_str04(self, df, col, new_col):
        try:
            df[new_col] = df[col].str[0:4].astype('Int64')
            self.append_scaling(new_col, 0)
        except Exception as e:
            print(e)
            print(f'Column {col} not found')
        return df
    
    def transformation_str05(self, df, col, new_col):
        try:
            df[new_col] = df[col].str[0:5]
        except Exception as e:
            print(e)
            print(f'Column {col} not found')
        return df
    def transformation_remove(self, df, col):
        try:
            df.drop(columns=[col], inplace=True)
        except Exception as e:
            print(e)
            print(f'Column {col} not found')
        return df
    
    def encoder_fit(pr, X):
        pr.encoder.fit(X)
        return pr

    def encoder_transform(pr, X):
        Xcolumns = X.columns
        X = pr.encoder.transform(X)
        X = pd.DataFrame(X.toarray(), columns=pr.encoder.get_feature_names_out(Xcolumns))
        return X
    
    def transformation_oneHot(self, df, col, new_col):
        try:
            self.encoder.fit(df[[col]])
            X = self.encoder.transform(df[[col]])
            X = pd.DataFrame(X.toarray(), columns=self.encoder.get_feature_names_out([col]))
            df = pd.concat([df, X], axis=1)
            for new_column in X.columns:
                self.append_scaling(new_column, 0)
            
        except Exception as e:
            print(e)
            print(f'Column {col} not found')
        return df
    
    def update_transformation_list(self,feature,method,new_col):
        i = 0
        for col, method, new_col in self.transformation_list:
            if col == feature:
                self.transformation_list[i] = (feature, method, new_col)
                return
            i += 1
            
    
    def add_transformation(self,feature,method,new_col):
        self.transformation_list.append((feature, method, new_col))
        

# --------------------------------- fill the missing values

    def update_fillna_list(self,df):
        self.fillna_list = [(col, 'median') for col in df.columns if df[col].dtype in ['Int64', 'int64', 'float64','Float64']]
        self.fillna_list.extend([(col, 'mode') for col in df.columns if df[col].dtype in ['string', 'object']])
        print(f'extended fillna_list: {self.fillna_list}')
    
    def fillna_pipeline(self,df):
        print(f'nans in the beginning: {df.isna().sum().sum()}')
        for col,method in self.fillna_list:
            if method == 'median':
                df = self.fillna_median(df, col)
            elif method == 'mode':
                df = self.fillna_mode(df, col)
            elif method == 'mean':
                df = self.fillna_mean(df, col)
            elif method == 'zero':
                df = self.fillna_zero(df, col)
            else:
                print(f'Unknown method {method} for column {col}')
            
            print(f'Column {col} is filled with {method} ->num nan: {df[col].isna().sum()}')

        print(f'nans in the end: {df.isna().sum().sum()}')

        return df

    def fillna_median(self, df, col):
        try:
            med = df[col].median().astype('int64')
            df[col].fillna(med, inplace=True)
        except Exception as e:
            print(e)
            print(f'Column {col} not found')
        return df
    
    def fillna_mode(self, df, col):
        try:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)
        except Exception as e:
            print(e)
            print(f'Column {col} not found')
        return df

    def fillna_mean(self, df, col):
        try:
            mean = df[col].mean().astype(df[col].dtype.name)
            df[col].fillna(mean, inplace=True)
        except Exception as e:
            print(e)
            print(f'Column {col} not found')
        return df

    def fillna_zero(self, df, col):
        try:
            df[col].fillna(0, inplace=True)
        except Exception as e:
            print(e)
            print(f'Column {col} not found')
        return df
    
    def look_for_fillna_method(self,feature):
        i = 0
        for col, method in self.fillna_list:
            if col == feature:
                print(f'-Column {col} is filled with {method} at index {i}')
                return method
            i += 1
        print(f'-Column {feature} not found in fillna list')
        return None
    
    def change_fillna_method(self,feature,method):
        i = 0
        for col, method in self.fillna_list:
            if col == feature:
                self.fillna_list[i] = (feature, method)
                return
            i += 1
        print(f'-Column {feature} not found in fillna list')
        self.add_fillna(feature, method)
        
    def add_fillna(self,feature,method):      
        self.fillna_list.append((feature, method))
# --------------------------------- scaling part
    def append_scaling(self,feature,yn):
        features = [col for col, yn in self.sclaing_list]
        if feature in features:
            print(f'Feature {feature} is already in scaling list')
            return
    
    
        if yn == 'Y'or yn == 'y' or yn == 'yes' or yn == 'Yes' or yn == 1:
            self.sclaing_list.append((feature,1))
        else:
            self.sclaing_list.append((feature,0))

    def update_scaling_list(self, feature, yn):
        i = 0
        for col, yn_ in self.sclaing_list:
            if col == feature:
                self.sclaing_list[i] = (feature, yn)
                return i
            i += 1
        print(f'Feature {feature} not found in scaling list, adding it now')
        self.append_scaling(feature, yn)
        return i
        
        
    
    def refresh_scaling_list(self,df):
        num_cols = [col for col in df.columns if df[col].dtype in ['Int64', 'int64', 'float64','Float64']]
        #other_cols = [col for col in df.columns if df[col] not in num_cols]
        cols_to_be_updated = set(df.columns) - set(self.sclaing_list)
        
        for col in cols_to_be_updated:
            if col in num_cols:
                self.append_scaling(col, 1)
            else:
                self.append_scaling(col, 0)
                
        
    def scaling_fit (self,sd):
        self.scaler.fit(sd)
        return self
    
    def scaling_transform(self,sd):
        sd = self.scaler.transform(sd)
        sd = pd.DataFrame(sd, columns = sd.columns)
        return sd

    def scaling_pipeline(self,sd):
        #self.refresh_scaling_list(sd)
        scaling_cols = [col for col, yn in self.sclaing_list if yn == 1]
        scaling_cols = [col for col in sd.columns if col in scaling_cols]
        other_cols = [col for col in sd.columns if col not in scaling_cols]
        if self.status == 'train':
            print(f'scaling_cols: {scaling_cols}')
            print(f'info: {sd[scaling_cols].info()}')
            self.scaler.fit(sd[scaling_cols])       
        X = self.scaler.transform(sd[scaling_cols])
        X = pd.DataFrame(X, columns = scaling_cols, index = sd.index)
        sd = pd.concat([sd[other_cols], X], axis=1)
        return sd

