import pandas as pd
import numpy as np

def cleanUp (df, to_csv = False):
    
    col_to_drop = []
    
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
    try: 
        df['Carrier Type Code_'] = df['Carrier Type'].str[:2]
        print("2. ---> Carrier Type encoded in Carrier Type Code_")
        col_to_drop.append('Carrier Type')
    except Exception as e:
        print(e)
        
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
    
    if to_csv:
        df.to_csv('../Data/cleaned_data.csv', index=False)
        print("----> saved the cleaned data")
    
    return df
    
    
    
         
    
    
    
                         
                        