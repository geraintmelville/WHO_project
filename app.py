#write functions here that are called by main.property
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

def li_pipeline(data):
    df = data
    X = df.drop('Life_expectancy', axis=1)
    y = df['Life_expectancy']
    
    # 2. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=X['Country']  # ← Stratify by country
    )

    # 3. Store country info if needed for analysis
    country_train = X_train['Country']
    country_test = X_test['Country']


    # 4. Drop non-needed categorical columns and OHE Region
    X_train = X_train.drop(['Country', 'Economy_status_Developed'], axis=1)
    X_test = X_test.drop(['Country', 'Economy_status_Developed'], axis=1)

    # One-hot encode Region (drop_first=True to avoid multicollinearity)
    X_train = pd.get_dummies(X_train, columns=['Region'], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=['Region'], drop_first=True)

    # Ensure test set has the same columns as train set
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # 5. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    # 7. Evaluate
    y_pred = model.predict(X_test_scaled)
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")

    # 8. Verify stratification worked
    print("\nCountry distribution preserved:")
    print(country_train.value_counts(normalize=True).head())
    print("\nTest set:")
    print(country_test.value_counts(normalize=True).head())

    return True


# least information model takes in non-sensitive
# data to make a prediction
def least_information(data):
    #create dataframe that only has the least_information columns
    df_li = data[[
    'Country',
    'Region',
    'Year',
    'Schooling',
    'GDP_per_capita',
    'Population_mln',
    'Economy_status_Developed'
    ]]
    
    return df_li
