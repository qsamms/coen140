import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA, KernelPCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel

# This function creates a dictionary mapping strings to a unique integer value
# For rows in the dataframe that are not integer or float values
def convertToNumerical(df):
    columns = df.columns.values
    for column in columns:
        vals = {}
        def convertInt(val):
            return vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            contents = df[column].values.tolist()
            unique = set(contents)
            x = 0
            for u in unique:
                if u not in vals:
                    vals[u] = x
                    x += 1
            df[column] = list(map(convertInt, df[column]))
    
trdf = pd.read_csv("train.dat")
convertToNumerical(trdf)
trdf2 = trdf.replace(np.nan,0.0)
print(trdf2.head())

X_train = trdf2.loc[:,trdf.columns != "TARGET"].to_numpy()
y_train = trdf2['TARGET'].to_numpy()

tstdf = pd.read_csv("test.dat")
convertToNumerical(tstdf)
tstdf2 = tstdf.replace(np.nan,0.0)
X_test = tstdf2.to_numpy()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

steps = [('pca',PCA(n_components=10)), ('linreg',LinearRegression())]
model = Pipeline(steps=steps)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

file = open("output_kpca.txt",'a')
for y in y_pred:
    formatted = '{:.1f}'.format(y)
    print(formatted)
    file.write(f'{formatted}\n')

