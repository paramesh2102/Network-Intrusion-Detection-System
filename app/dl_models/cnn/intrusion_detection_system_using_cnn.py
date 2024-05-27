#from tensorflow.keras import Conv1D,MaxPooling1D,BatchNormalization,Activation

from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,Conv1D,BatchNormalization,Activation
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
#from keras.api.layers import MaxPooling1D
import os
from keras.layers import MaxPooling1D



import pandas as pd
import numpy as np
import sys
import sklearn.preprocessing
import keras
import sklearn
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Embedding, Flatten

#from keras import Sequential,Model.Sequential
#import keras.Model.Sequential

#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
#from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization,Convolution1D,MaxPooling1D, Reshape, GlobalAveragePooling1D
#from keras.utils import to_categorical

from sklearn import metrics
from scipy.stats import zscore
#from tensorflow.keras.utils import get_file, plot_model
#from keras.utils import get_file,plot_model
#from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
#from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
print(pd.__version__)
print(np.__version__)
print(sys.version)
print(sklearn.__version__)

#Loading training set into dataframe
df = pd.read_csv('NSL-KDD/KDDTrain+.txt', header=None)
df.head()

#Loading testing set into dataframe
qp = pd.read_csv('NSL-KDD/KDDTest+.txt', header=None)
qp.head()

#Reset column names for training set
df.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
'num_access_files', 'num_outbound_cmds', 'is_host_login',
'is_guest_login', 'count', 'srv_count', 'serror_rate',
'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
'dst_host_srv_rerror_rate', 'subclass', 'difficulty_level']
df.head()

#Reset column names for testing set
qp.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
'num_access_files', 'num_outbound_cmds', 'is_host_login',
'is_guest_login', 'count', 'srv_count', 'serror_rate',
'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
'dst_host_srv_rerror_rate', 'subclass', 'difficulty_level']
qp.head()

#accessing names of training columns
lst_names = df.columns # returns a list of column names
lst_names

#accessing names of testing columns
testlst_names = qp.columns
testlst_names

#Dropping the last columns of training set
#df = df.drop('difficulty_level', 1) # we don't need it in this project
df=df.drop('difficulty_level',1)
df.shape

#Dropping the last columns of testing set
qp = qp.drop('difficulty_level', axis=1)
qp.shape

df.isnull().values.any()

qp.isnull().values.any()

#defining col list for one hot encoding
cols = ['protocol_type','service','flag']
cols

#One-hot encoding
def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(each,1)
    return df

#Merging train and test data
combined_data = pd.concat([df,qp])

combined_data.head(2)

import pandas as pd

combined_data.to_csv('data.csv', index=False)  # Set index=False to exclude the index column from the CSV

combined_data['subclass'].value_counts()





#Applying one hot encoding to combined data
combined_data = one_hot(combined_data,cols)

fc=['dst_host_srv_serror_rate',
 'service_ecr_i',
 'flag_RSTO',
 'service_urh_i',
 'flag_OTH',
 'dst_host_serror_rate',
 'diff_srv_rate',
 'dst_host_same_src_port_rate',
 'serror_rate',
 'flag_RSTOS0',
 'wrong_fragment',
 'protocol_type_icmp',
 'logged_in',
 'srv_serror_rate',
 'dst_host_same_srv_rate',
 'flag_RSTR',
 'is_host_login',
 'is_guest_login',
 'srv_diff_host_rate',
 'service_eco_i',
 'flag_REJ',
 'flag_S0',
 'service_red_i',
 'dst_host_srv_count',
 'count',
 'same_srv_rate',
 'service_pop_3',
 'protocol_type_udp',
 'dst_host_srv_diff_host_rate',
 'flag_SF',
 'srv_count',
 'dst_host_diff_srv_rate',
 'flag_S3',
 'num_failed_logins',
 'land',
 'flag_SH',
 'flag_S2',
 'flag_S1',
 'service_urp_i',
 'protocol_type_tcp',
 'service_ftp',
    'subclass']

len(fc)

combined_data=combined_data[fc]

#Function to min-max normalize
def normalize(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with normalized specified features
    """
    result = df.copy() # do not touch the original df
    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if max_value > min_value:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

#Dropping subclass column for training set
#tmp = combined_data.pop('subclass')

#tmp

temp_target=combined_data.pop('subclass')

temp_target

combined_data.tail(2)



#Normalizing training set
new_train_df = normalize(combined_data,combined_data.columns)
new_train_df


#classlist

new_train_df.rename(columns={'subclass': 'Class'}, inplace=True)

#Appending class column to training set
new_train_df["Class"] = temp_target
new_train_df

new_train_df["Class"].value_counts()

len(new_train_df["Class"].value_counts())

new_train_df.isnull().values.any()

y_train=new_train_df["Class"]
y_train

y_train.isnull().values.any()

temp_dummies=pd.get_dummies(y_train)

temp_dummies.head(2)

temp_dummies.values

combined_data_X = new_train_df.drop('Class', 1)
combined_data_X



oos_pred = []

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
kfold.get_n_splits(combined_data_X,y_train)

#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Embedding, Flatten,Conv1D

#!pip uninstall keras
#!pip uninstall tensorflow
#!pip install keras
#!pip install tensorflow
#from keras.models import Sequential
#from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten, Dropout, Dense, Activation
from tensorflow.keras.optimizers import Adam

y_train

combined_data.head(2)

y_train

#len(y_test_2)



y_train


temp_dummies











len(y_train.value_counts())

pd.get_dummies(y_train).shape[1]

new_train_df.head(2)

combined_data_X = new_train_df.drop('Class', 1)
combined_data_X



from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

# Apply one-hot encoding to the entire target variable before splitting
y_encoded = pd.get_dummies(y_train)  # Assuming y_train is your target variable

# Define the number of classes based on the one-hot encoding
num_classes = y_encoded.shape[1]

# Split the data into a single train/test split
train_X, test_X, train_y, test_y = train_test_split(combined_data_X, y_encoded, test_size=0.2, random_state=42)

x_columns_train = new_train_df.columns.drop('Class')
x_train_array = train_X[x_columns_train].values
x_train_1 = np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1))

x_columns_test = new_train_df.columns.drop('Class')
x_test_array = test_X[x_columns_test].values
x_test_2 = np.reshape(x_test_array, (x_test_array.shape[0], x_test_array.shape[1], 1))

# Create the model with the updated output layer (num_classes dynamically determined)
model = Sequential()
model.add(Conv1D(64, kernel_size=15, padding="same", input_shape=(41, 1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(pool_size=5))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))  # Updated output layer

# Define your optimizer
optimizer = Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model on the training data
model.fit(x_train_1, train_y, epochs=500)

# Evaluate the model on the test data
pred = model.predict(x_test_2)
pred = np.argmax(pred, axis=1)
y_eval = np.argmax(test_y.values, axis=1)

# Calculate and print the test accuracy
test_accuracy = metrics.accuracy_score(y_eval, pred)
print("Test Accuracy: {:.2f}".format(test_accuracy))



#pd.get_dummies(y_train).columns



#oos_pred

#dummies_test.columns

len(x_test_2)

len(pred)

mindf=pd.read_csv('mindf.csv')

maxdf=pd.read_csv('maxdf.csv')

mindf

maxdf

new_data = np.array([[111,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,123,1,1,1]])

inputvar=['dst_host_srv_serror_rate', 'service_ecr_i', 'flag_RSTO',
       'service_urh_i', 'flag_OTH', 'dst_host_serror_rate', 'diff_srv_rate',
       'dst_host_same_src_port_rate', 'serror_rate', 'flag_RSTOS0',
       'wrong_fragment', 'protocol_type_icmp', 'logged_in', 'srv_serror_rate',
       'dst_host_same_srv_rate', 'flag_RSTR', 'is_host_login',
       'is_guest_login', 'srv_diff_host_rate', 'service_eco_i', 'flag_REJ',
       'flag_S0', 'service_red_i', 'dst_host_srv_count', 'count',
       'same_srv_rate', 'service_pop_3', 'protocol_type_udp',
       'dst_host_srv_diff_host_rate', 'flag_SF', 'srv_count',
       'dst_host_diff_srv_rate', 'flag_S3', 'num_failed_logins', 'land',
       'flag_SH', 'flag_S2', 'flag_S1', 'service_urp_i', 'protocol_type_tcp',
       'service_ftp']

newdf = pd.DataFrame(new_data, columns=inputvar)

newdf

targetdf = (newdf - mindf) / (maxdf - mindf)

targetdf

np.array(targetdf)[0]

l=[]

for i in np.array(targetdf)[0]:
  l.append([i])

l

l=np.array([l])

['dst_host_srv_serror_rate', 'service_ecr_i', 'flag_RSTO',
       'service_urh_i', 'flag_OTH', 'dst_host_serror_rate',
  'diff_srv_rate','dst_host_same_src_port_rate', 'serror_rate',
 'flag_RSTOS0','wrong_fragment', 'protocol_type_icmp',
 'logged_in', 'srv_serror_rate','dst_host_same_srv_rate',
 'flag_RSTR', 'is_host_login','is_guest_login',
 'srv_diff_host_rate', 'service_eco_i', 'flag_REJ',



 'flag_S0', 'service_red_i', 'dst_host_srv_count'
  'count','same_srv_rate', 'service_pop_3',
 'protocol_type_udp','dst_host_srv_diff_host_rate', 'flag_SF',
 'srv_count','dst_host_diff_srv_rate', 'flag_S3',
 'num_failed_logins', 'land','flag_SH',
 'flag_S2', 'flag_S1', 'service_urp_i',
 'protocol_type_tcp','service_ftp']

a=np.array([[[1],[1],[1],
[0],[0],[1],
[1],[1],[1],
[0],[1],[1],
[1],[1],[1],
[0],[1],[1],
[1],[0],[0],

[0],[0],[1],
[1],[1],[0],
[0],[1],[0],
[1],[1],[0],
[1],[1],[0],
[0],[0],[0],[0],[0]]])

col_list=['apache2', 'back', 'buffer_overflow', 'ftp_write', 'guess_passwd',
       'httptunnel', 'imap', 'ipsweep', 'land', 'loadmodule', 'mailbomb',
       'mscan', 'multihop', 'named', 'neptune', 'nmap', 'normal', 'perl',
       'phf', 'pod', 'portsweep', 'processtable', 'ps', 'rootkit', 'saint',
       'satan', 'sendmail', 'smurf', 'snmpgetattack', 'snmpguess', 'spy',
       'sqlattack', 'teardrop', 'udpstorm', 'warezclient', 'warezmaster',
       'worm', 'xlock', 'xsnoop', 'xterm']

new_data = np.array([[1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0,

                                  0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]])
newdf = pd.DataFrame(new_data, columns=inputvar)
targetdf = (newdf - mindf) / (maxdf - mindf)
l = []
for i in np.array(targetdf)[0]:
  l.append([i])
l = np.array([l])
pr2 = model.predict(l)
pr2 = np.argmax(pr2, axis=1)
print(pr2[0])
prediction_result=col_list[pr2[0]]
print('prediction using cnn is ', prediction_result)

pr2=model.predict(a)
pr2 = np.argmax(pr2,axis=1)
print(pr2[0])

pr2=model.predict(l)
pr2 = np.argmax(pr2,axis=1)
print(pr2[0])

import pickle
pickle.dump(model, open('attack_type_model_one_hot_using_cnn.pkl','wb'))

l=[0,0,0]
#l.index(1)
max(l)
l.index(max(l))

