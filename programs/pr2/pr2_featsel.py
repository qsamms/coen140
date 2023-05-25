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

cor = trdf2.corr()
print(cor)
cor_target = abs(cor["TARGET"])
cor_target2 = cor_target[cor_target>0.01]
relevant_features_df = cor_target2.to_frame()
relevant_features_df.rename(columns = {0: 'TARGET'})
relevant_features_df.sort_values(
    by='TARGET',
    ascending=False
)

print(relevant_features_df.head())
print("relevnat feature shape: :", relevant_features_df.shape)
relevant_column_list = list(relevant_features_df.index)
rel_columns = relevant_column_list[:len(relevant_column_list)-1]
print("# relevant columns:",len(rel_columns))

X_tr = trdf2.loc[:,trdf.columns != "TARGET"]
X_train = X_tr.loc[:,rel_columns].to_numpy()
y_train = trdf2['TARGET'].to_numpy()

tstdf = pd.read_csv("test.dat")
convertToNumerical(tstdf)
tstdf2 = tstdf.replace(np.nan,0.0)
X_test = tstdf2.loc[:,rel_columns].to_numpy()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

steps = [('pca',PCA(n_components=10)), ('linreg',LinearRegression())]
model = Pipeline(steps=steps)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

file = open("output_featSel_PCA.txt",'a')
for y in y_pred:
    formatted = '{:.1f}'.format(y)
    file.write(f'{formatted}\n')

