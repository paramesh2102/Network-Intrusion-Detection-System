

import numpy as np
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import numpy as np
from sklearn import metrics
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten, Dropout, Dense, Activation,Embedding
from keras.optimizers import Adam
from joblib import dump, load
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,BaggingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix,roc_curve, auc
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from keras.models import Sequential
import pandas as pd
import numpy as np
import sys
import keras
import sklearn
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten,LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization,Convolution1D,MaxPooling1D, Reshape, GlobalAveragePooling1D
from keras.utils import to_categorical
import sklearn.preprocessing
from sklearn import metrics
from scipy.stats import zscore
from keras.utils import get_file,plot_model
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
lst_names = df.columns

testlst_names = qp.columns
testlst_names

df = df.drop('difficulty_level', 1)


qp = qp.drop('difficulty_level', 1)

df.isnull().values.any()
qp.isnull().values.any()

#defining col list for one hot encoding
cols = ['protocol_type','service','flag']
#One-hot encoding
def one_hot(df, cols):
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(each, 1)
    return df

#Merging train and test data
combined_data = pd.concat([df,qp])
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
            result[feature_name] =(df[feature_name] - min_value) / (max_value - min_value)
    return result

temp_target=combined_data.pop('subclass')
#finding min ,max values for normalization
min_values=combined_data.min()
max_values=combined_data.max()
#Normalizing training set
new_train_df = normalize(combined_data,combined_data.columns)
new_train_df.rename(columns={'subclass': 'Class'}, inplace=True)
#Appending class column to training set
new_train_df["Class"] = temp_target
len(new_train_df["Class"].value_counts())
new_train_df.isnull().values.any()
y_train=new_train_df["Class"]
y_train.isnull().values.any()
pd.get_dummies(y_train).head(2)
y_encoded=pd.get_dummies(y_train)
combined_data_X = new_train_df.drop('Class', 1)
combined_data_X.head(2)

X=combined_data_X

y_train.head(2)

y=y_train

y

# One-hot encode the target variable
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

y_encoded

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

X_train_normalized=X_train
X_test_normalized=X_test

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=41, output_dim=40, input_length=41),
    tf.keras.layers.LSTM(units=64),
    tf.keras.layers.Dense(40, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_normalized, y_train, epochs=1, batch_size=32, validation_split=0.2)

accuracy = model.evaluate(X_test_normalized, y_test, verbose=0)[1]
print(f"Test Accuracy: {accuracy * 100:.2f}%")

mindf=pd.read_csv('mindf.csv')
maxdf=pd.read_csv('maxdf.csv')

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
col_list=['apache2', 'back', 'buffer_overflow', 'ftp_write', 'guess_passwd',
       'httptunnel', 'imap', 'ipsweep', 'land', 'loadmodule', 'mailbomb',
       'mscan', 'multihop', 'named', 'neptune', 'nmap', 'normal', 'perl',
       'phf', 'pod', 'portsweep', 'processtable', 'ps', 'rootkit', 'saint',
       'satan', 'sendmail', 'smurf', 'snmpgetattack', 'snmpguess', 'spy',
       'sqlattack', 'teardrop', 'udpstorm', 'warezclient', 'warezmaster',
       'worm', 'xlock', 'xsnoop', 'xterm']

new_data = np.array([[1,0,1,1,1,1,1,1,1,1,20,1,1,1,1
                      ,0,1,11,1,1,1,1,13,14,14,1,1,1,1,1,
                      0,1,1,11,11,12,1,19,18,17,16]])
newdf=pd.DataFrame(new_data,columns=inputvar)
new_data=(newdf-mindf)/(maxdf-mindf)

new_data.head(2)

#new_data_normalized = normalize(new_data, columns_to_normalize)
predictions = model.predict(new_data)
predicted_classes = np.argmax(predictions, axis=1)

predictions

result=col_list[predicted_classes[0]]
result


import pickle
pickle.dump(model, open('attack_type_model_one_hot_using_rnn.pkl','wb'))
#model=pickle.load(open('attack_type_model_one_hot.pkl', 'rb'))