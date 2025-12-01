import pandas as pd
from MultipleRegression import regress
from Grapher import plotMultipleRegression as plot
from Grapher import linearToString
from sklearn.preprocessing import MinMaxScaler

# Read the dataset in
insurance_df = pd.read_csv("insurance_dataset.csv")

# Drop all smokers
insurance_df = insurance_df.drop(insurance_df[insurance_df.Smoker == "yes"].index)

# Drop categroical variables
df = insurance_df.drop(['Smoker', 'Number_of_Children', 'Gender', 'Region', ], axis=1) 
# New dataframe Has ["Age", "BMI", "Insurance_Cost"] for nonsmokers

scaler = MinMaxScaler()
df_normalized_minmax = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 2 Independent Variables
X = df_normalized_minmax[['Age', 'BMI']].to_numpy()
# 1 Dependent Variable
Y = df_normalized_minmax["Insurance_Cost"].to_numpy()

# Regress, print, and plot
R = regress(X, Y)
print(linearToString(R))
plot(X, Y, R)