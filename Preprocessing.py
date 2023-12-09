from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def get_train_test(regression=True):
    # Get the data
    DATA = Path("data") 
    eq_raw_df = pd.read_csv(DATA / "nepal-earthquake-severity-index-latest.csv", 
                            low_memory=False)
    
    # Delete the uninteresting data
    columns = ['REGION', 'Hazard (Intensity)',
               'Exposure', 'Housing', 'Poverty', 
               'Vulnerability','Severity','Severity category']
    eq_df = eq_raw_df[columns].copy()

    # Rename columns
    rename_dict = {
        "Hazard (Intensity)": "INTENSITY",
        "Severity category": "SEVERITY_CATEGORY",
        "Severity" : "SEVERITY",
        "Exposure" : "EXPOSURE",
        "Housing" : "HOUSING",
        "Poverty" : "POVERTY",
        "Vulnerability" : "VULNERABILITY"
    }

    eq_df.rename(columns=rename_dict, inplace=True)
    del eq_raw_df

    # Drop the missing values
    eq_df.dropna(axis="index", inplace=True)
    eq_df.reset_index(drop=True, inplace=True)

    # perform one-hot encoding on categorical feature
    if(regression):
        categorical_features = ['REGION']
    else:
        categorical_features = ['REGION','SEVERITY_CATEGORY']
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(eq_df[categorical_features])
    df_features_encoded = pd.DataFrame(enc.transform(eq_df[categorical_features]).toarray(), columns=enc.get_feature_names_out())
    # combine the one-hot encoded features with the numerical features
    eq_df = pd.concat([eq_df.drop(categorical_features, axis=1), df_features_encoded ], axis=1)

    # Split the data into train and test sets
    if(regression):
        X = eq_df.drop(columns=['SEVERITY','SEVERITY_CATEGORY'])
        y = eq_df['SEVERITY']
    else:
        X = eq_df.drop(columns=['SEVERITY','SEVERITY_CATEGORY_High',
                                'SEVERITY_CATEGORY_Highest', 'SEVERITY_CATEGORY_Low',
                                'SEVERITY_CATEGORY_Lowest', 'SEVERITY_CATEGORY_Medium-High',
                                'SEVERITY_CATEGORY_Medium-Low'])
        y = eq_df['SEVERITY_CATEGORY_High','SEVERITY_CATEGORY_Highest', 
                  'SEVERITY_CATEGORY_Low','SEVERITY_CATEGORY_Lowest', 
                  'SEVERITY_CATEGORY_Medium-High','SEVERITY_CATEGORY_Medium-Low']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test