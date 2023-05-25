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

#read in the training data and preprocess it
trdf = pd.read_csv("train.dat")
convertToNumerical(trdf)
trdf2 = trdf.replace(np.nan,0.0)
print(trdf2.head())

#find correlation values between the features and target variable
cor = trdf2.corr()
cor_target = abs(cor["TARGET"])
#filter out low coorelations
cor_target2 = cor_target[cor_target>0.01]
relevant_features_df = cor_target2.to_frame()
relevant_features_df.rename(columns = {0: 'TARGET'})
relevant_features_df.sort_values(
    by='TARGET',
    ascending=False
)

print("relevnat feature shape: :", relevant_features_df.shape)
relevant_column_list = list(relevant_features_df.index)
rel_columns = relevant_column_list[:len(relevant_column_list)-1]
print("# relevant columns:",len(rel_columns))

#get training set as two numpy arrays, only columns of interest
X_tr = trdf2.loc[:,trdf.columns != "TARGET"]
X_train = X_tr.loc[:,rel_columns].to_numpy()
y_train = trdf2['TARGET'].to_numpy()

#read in test set, preprocess it and set it as a numpy array, only columns of interest
tstdf = pd.read_csv("test.dat")
convertToNumerical(tstdf)
tstdf2 = tstdf.replace(np.nan,0.0)
X_test = tstdf2.loc[:,rel_columns].to_numpy()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

#pipeline to train and test the model 
steps = [('pca',PCA(n_components=5)), ('linreg',LinearRegression())]
model = Pipeline(steps=steps)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

#format output
file = open("output_featSel_PCA-5.txt",'a')
for y in y_pred:
    formatted = '{:.1f}'.format(y)
    file.write(f'{formatted}\n')

