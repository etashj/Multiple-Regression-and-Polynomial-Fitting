import pandas as pd
from MultipleRegression import regress
from Grapher import plotMultipleRegression as plot
from Grapher import linearToString
from sklearn.preprocessing import MinMaxScaler

# Read the dataset in
insurance_df = pd.read_csv("insurance_dataset.csv")

# Drop all smokers
# insurance_df = insurance_df.drop(
#     insurance_df[insurance_df.Smoker == "yes"].index
# )

# Keep only smokers
insurance_df = insurance_df.drop(
    insurance_df[insurance_df.Smoker == "no"].index
)

# Drop categroical variables
df = insurance_df.drop(['Smoker', 
                        'Number_of_Children', 
                        'Gender', 
                        'Region', ],
                    axis=1) 
# New dataframe Has ["Age", "BMI", "Insurance_Cost"] for nonsmokers

scaler = MinMaxScaler()
df_normalized_minmax = pd.DataFrame(
    scaler.fit_transform(df), 
    columns=df.columns
)

# 2 Independent Variables
X = df_normalized_minmax[['Age', 'BMI']].to_numpy()
# 1 Dependent Variable
Y = df_normalized_minmax["Insurance_Cost"].to_numpy()

# Regress, print, and plot
R = regress(X, Y)
print(linearToString(R))
plot(X, Y, R)

### Results ###
'''
Nonsmoker:
y = 0.33910220484082676 
    + 0.08830935338117211(x_1) 
    + 0.28065510020236023(x_2)

(NORM_INSURANCE) = 0.33910220484082676
                   + 0.08830935338117211(NORM_AGE)
                   + 0.28065510020236023(NORM_BMI)

Smoker: 
y = 0.38916000594004924 
    + 0.170081177548804(x_1) 
    + 0.23808026071054747(x_2)

(NORM_INSURANCE) = 0.38916000594004924 
    + 0.170081177548804(NORM_AGE)
    + 0.23808026071054747(NORM_BMI)


Note to use this equation you mut normalize the age and BMI, 
    then undo the normalization of insurance. 
This will present itself as a composition of several functions. 
'''