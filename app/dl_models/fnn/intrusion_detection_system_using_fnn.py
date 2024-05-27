import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
from keras.models import Sequential

from keras.layers import Dense,Dropout,Activation,Embedding,Flatten,Conv1D
#from keras.layers import Dense, Dropout, Activation, Embedding, Flatten,Conv1D
from keras.models import Sequential
#from keras.models import Sequential
#from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten, Dropout, Dense, Activation
from keras.optimizers import Adam
from joblib import dump, load
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
#from scipy import interp
from itertools import cycle
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
import pandas as pd
import numpy as np
import sys
import keras
import sklearn
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
#from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization,Convolution1D,MaxPooling1D, Reshape, GlobalAveragePooling1D

from keras.utils import to_categorical
import sklearn.preprocessing
from sklearn import metrics
from scipy.stats import zscore
from keras.utils import get_file,plot_model
#from keras.api.cal
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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
df = df.drop('difficulty_level',1) # we don't need it in this project
#df = df.drop(columns=['difficulty_level'])
print(df.head(2))
df.shape

#Dropping the last columns of testing set
qp = qp.drop('difficulty_level',1)
#qp = qp.drop(columns=['difficulty_level'])
qp.shape

df.isnull().values.any()

qp.isnull().values.any()

#defining col list for one hot encoding
cols = ['protocol_type','service','flag']
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
    print('dummy data frame is ',df)
    return df

#Merging train and test data
combined_data = pd.concat([df,qp])

combined_data.head(2)

"""
combined_data=combined_data[['duration', 'protocol_type', 'src_bytes',
       'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
       'su_attempted','subclass']]
"""

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

combined_data=combined_data[fc]

combined_data.head(2)

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
            print(df)
            print(min_value)
            print(max_value)
            # Ensure that df[feature_name], min_value, and max_value are numeric arrays or values
            # For example, if they are DataFrame columns:
            feature_values = df[feature_name].values
            min_value = df[feature_name].min()
            max_value = df[feature_name].max()



            result[feature_name] =(df[feature_name] - min_value) / (max_value - min_value)
    return result

temp_target=combined_data.pop('subclass')

combined_data.head(5)

normalize(combined_data,combined_data.columns)

combined_data.head(2)



#finding min ,max values for normalization
min_values=combined_data.min()



max_values=combined_data.max()







len(min_values)

len(max_values)

min_values

max_values

combined_data.head(2)

#Normalizing training set
new_train_df = normalize(combined_data,combined_data.columns)
new_train_df

new_train_df.rename(columns={'subclass': 'Class'}, inplace=True)

#Appending class column to training set
new_train_df["Class"] = temp_target
new_train_df

len(new_train_df["Class"].value_counts())



new_train_df.isnull().values.any()

y_train=new_train_df["Class"]
y_train

y_train.isnull().values.any()

pd.get_dummies(y_train).head(2)

y_encoded=pd.get_dummies(y_train)

y_encoded.head(2)

combined_data_X = new_train_df.drop('Class', 1)

combined_data_X

# Split the data into a single train/test split
train_X, test_X, train_y, test_y = train_test_split(combined_data_X, y_encoded, test_size=0.2, random_state=42)

train_X.head(2).columns

fl=list(train_X.head(2).columns)

test_X.head(2)

train_y.head(2).columns

test_y.head(2)

# Create a Feedforward Neural Network (FNN) model
print('before model')
model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', random_state=42)
#model = MLPClassifier(c=(4, 2), activation='relu', solver='adam', random_state=42)
print('after model')
# Train the model on the training data
model.fit(train_X, train_y)
print('after fit')

# Evaluate the model on the test data
accuracy = model.score(test_X, test_y)
print(f"Test Accuracy: {accuracy * 100:.2f}%")





len(fl)

fl

inputvar=fl

new_data = np.array([[1,0,1,1,1,1,1,1,1,1,20,1,1,1,1
                      ,0,1,11,1,1,1,1,13,14,14,1,1,1,1,1,
                      0,1,1,11,11,12,1,19,18,17,16]])

len(new_data[0])

newdf=pd.DataFrame(new_data,columns=inputvar)
newdf

#normalize the newdf data frame

max_values

min_values

mindf=pd.DataFrame(min_values).T

mindf

maxdf=pd.DataFrame(max_values).T

maxdf

mindf.to_csv('mindf.csv')
maxdf.to_csv('maxdf.csv')



targetdf=(newdf-mindf)/(maxdf-mindf)

targetdf.head(2)

targetdf.values

new_data = targetdf.values

prediction = model.predict(new_data)
print(f"Prediction: {prediction}")

#result=list(prediction[0]).index(1)
col_list=['apache2', 'back', 'buffer_overflow', 'ftp_write', 'guess_passwd',
       'httptunnel', 'imap', 'ipsweep', 'land', 'loadmodule', 'mailbomb',
       'mscan', 'multihop', 'named', 'neptune', 'nmap', 'normal', 'perl',
       'phf', 'pod', 'portsweep', 'processtable', 'ps', 'rootkit', 'saint',
       'satan', 'sendmail', 'smurf', 'snmpgetattack', 'snmpguess', 'spy',
       'sqlattack', 'teardrop', 'udpstorm', 'warezclient', 'warezmaster',
       'worm', 'xlock', 'xsnoop', 'xterm']
pr2 = np.argmax(prediction, axis=1)
result =pr2[0]
result=col_list[result]

import random
l=[i for i in range(0,41)]
try:
  result=list(prediction[0]).index(1)
except:
  result=random.choice(l)

#exporting the model
import pickle
pickle.dump(model, open('attack_type_model_one_hot.pkl','wb'))
model=pickle.load(open('attack_type_model_one_hot.pkl', 'rb'))



