import pickle
import werkzeug.security

# Print out the supported hash methods
#print("Supported hash methods:", werkzeug.security.)

from flask import Flask, render_template, request, redirect, url_for,session
from flask_sqlalchemy import SQLAlchemy
#import SQLAlchemy

import sqlalchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.security import generate_password_hash, check_password_hash
import random,os,base64,smtplib,pyotp,os
import pandas as pd
import numpy as np
from flask import jsonify
import joblib
import subprocess
import csv
#import ydata_profiling
#import ydata_profiling
import ydata_profiling

#import pandas_profiling
from pathlib import Path;
#import pandas_profiling
#import smtplib
#import pyotp
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend


app = Flask(__name__, static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone_number = db.Column(db.String(20), nullable=False)


'''
import keras
import tensorflow as tf
print("Keras version:", keras.__version__)
print("TensorFlow version:", tf.__version__)
'''


@app.route('/')
def index():
    return render_template('index.html')




# Set a secret key for session management
app.secret_key = 'your_secret_key_here'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            # Set a session variable to mark the user as logged in
            session['logged_in'] = True
            return redirect(url_for('home'))

    return render_template('login.html')




@app.route('/home', methods=['GET', 'POST'])
def home():
    # Check if the user is logged in before rendering the home page
    if session.get('logged_in'):
        return render_template('home.html')
    else:
        return redirect(url_for('login'))



#to solve the error of binascii.Error: Non-base32 digit found


otp_secret = base64.b32encode(os.urandom(20)).decode('utf-8')
print("Generated OTP Secret Key:", otp_secret)
#to solve the above error


otp_secret = otp_secret
otp = pyotp.TOTP(otp_secret)





# Function to send OTP email using Gmail's SMTP server
def send_otp_email(email, otp):
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'lchaitanya2003@gmail.com'  # Your Gmail email address
    smtp_password = 'fqxpzvfcrbykmbtr'  # Your Gmail app password

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        message = f'Subject: To sign up into Intrusion Detection System\n\nYour OTP: {otp}'
        server.sendmail(smtp_username, email, message)
        server.quit()
        return True
    except Exception as e:
        print("Email sending error:", e)
        return False








@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        name = request.form.get('name')
        email = request.form.get('email')
        phone_number = request.form.get('phone_number')

        # Store user details in session for later use
        session['signup_username'] = username
        session['signup_password'] = password
        session['signup_name'] = name
        session['signup_email'] = email
        session['signup_phone_number'] = phone_number

        generated_otp = otp.now()  # Generate OTP

        # Send OTP via email using smtplib
        send_otp_email(email, generated_otp)

        session['generated_otp'] = generated_otp
        return redirect(url_for('verify_otp'))

    return render_template('signup.html')


@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        user_otp = request.form.get('otp')

        if user_otp == session['generated_otp']:
            # OTP verified, create account and add to the database
            username = session['signup_username']
            password = session['signup_password']
            name = session['signup_name']
            email = session['signup_email']
            phone_number = session['signup_phone_number']

            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = User(username=username, password=hashed_password, name=name, email=email, phone_number=phone_number)

            db.session.add(new_user)
            db.session.commit()

            del session['generated_otp']  # Clear session data
            del session['signup_username']
            del session['signup_password']
            del session['signup_name']
            del session['signup_email']
            del session['signup_phone_number']

            return render_template('account_created.html', username=username, password=password)
        else:
            return "Invalid OTP"

    return render_template('verify_otp.html')








@app.route('/upload', methods=['POST'])
def upload_file():
    nameofreport=''
    if 'dataset' not in request.files:
        upload_status = "No selected file"
    else:
        file = request.files['dataset']
        if file.filename == '':
            upload_status = "You have not  selected any file"
        else:
            # Save the uploaded file
            file.save(os.path.join('static/uploads', file.filename))

            upload_status = "File uploaded successfully"
            data = pd.read_csv(os.path.join('static/uploads', file.filename))
            profile_report = ydata_profiling.ProfileReport(data)


            nameoffile=Path(os.path.join('static/uploads', file.filename)).stem
            nameofreport=nameoffile+'_report.html'
            profile_report.to_file('static/uploads/'+nameofreport)

    return render_template('home.html', upload_status=upload_status,name_of_report=nameofreport)



@app.route('/csv_to_html')
def csv_to_html():
    csv_path = 'static/uploads/KDDTrain+_20Percent.csv'
    table_html = '<table border="1">'
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            table_html += '<tr>'
            for cell in row:
                table_html += f'<td>{cell}</td>'
            table_html += '</tr>'
    table_html += '</table>'
    return render_template('csv_to_html.html', table_html=table_html)




"""
@app.route('/nids_website/app/Network_Intrusion_Detection_System/app1.py', methods=['GET'])
def run_nids_app():
    return render_template('index_nids.html')"""

import pandas as pd






































"""
@app.route('/nids')
def nids():
    return render_template('index_nids.html')
    


model = joblib.load('/nids_website/app/Network_Intrusion_Detection_System/model.pkl')
@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]

    if int_features[0]==0:
        f_features=[0,0,0]+int_features[1:]
    elif int_features[0]==1:
        f_features=[1,0,0]+int_features[1:]
    elif int_features[0]==2:
        f_features=[0,1,0]+int_features[1:]
    else:
        f_features=[0,0,1]+int_features[1:]

    if f_features[6]==0:
        fn_features=f_features[:6]+[0,0]+f_features[7:]
    elif f_features[6]==1:
        fn_features=f_features[:6]+[1,0]+f_features[7:]
    else:
        fn_features=f_features[:6]+[0,1]+f_features[7:]

    final_features = [np.array(fn_features)]

    predict = model.predict(final_features)

    if predict==0:
        output='Normal'
    elif predict==1:
        output='DOS'
    elif predict==2:
        output='PROBE'
    elif predict==3:
        output='R2L'
    else:
        output='U2R'

    return render_template('index_nids.html', output=output)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    predict = model.predict([np.array(list(data.values()))])

    if predict==0:
        output='Normal'
    elif predict==1:
        output='DOS'
    elif predict==2:
        output='PROBE'
    elif predict==3:
        output='R2L'
    else:
        output='U2R'

    return jsonify(output)"""














@app.route("/features")
def features():
    return render_template("features.html")


@app.route("/dlmodels")
def dlmodels():
    return render_template("dlmodels.html")


@app.route("/mlmodels")
def mlmodels():
    return render_template("mlmodels.html")





"""
col_list=['apache2', 'back', 'buffer_overflow', 'ftp_write', 'guess_passwd',
       'httptunnel', 'imap', 'ipsweep', 'land', 'loadmodule', 'mailbomb',
       'mscan', 'multihop', 'named', 'neptune', 'nmap', 'normal', 'perl',
       'phf', 'pod', 'portsweep', 'processtable', 'ps', 'rootkit', 'saint',
       'satan', 'sendmail', 'smurf', 'snmpgetattack', 'snmpguess', 'spy',
       'sqlattack', 'teardrop', 'udpstorm', 'warezclient', 'warezmaster',
       'worm', 'xlock', 'xsnoop', 'xterm']
       
"""


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


@app.route('/predict_dlmodels', methods=['GET', 'POST'])
def predict_dlmodels():
    if request.method == 'POST':
        var1 = request.form.get('var1')
        var2 = request.form.get('var2')
        var3 = request.form.get('var3')
        var4 = request.form.get('var4')
        var5 = request.form.get('var5')
        var6 = request.form.get('var6')
        var7 = request.form.get('var7')
        var8 = request.form.get('var8')
        var9 = request.form.get('var9')
        var10 = request.form.get('var10')
        var11 = request.form.get('var11')
        var12 = request.form.get('var12')
        var13 = request.form.get('var13')
        var14 = request.form.get('var14')
        var15 = request.form.get('var15')
        var16 = request.form.get('var16')
        var17 = request.form.get('var17')
        var18 = request.form.get('var18')
        var19 = request.form.get('var19')
        var20 = request.form.get('var20')
        var21 = request.form.get('var21')
        var22 = request.form.get('var22')
        var23 = request.form.get('var23')
        # 'dst_host_srv_serror_rate'
        v1 = float(var1)
        # 'service_ecr_i'
        if (float(var2) == 1):
            v2 = 1
        else:
            v2 = 0
        # 'flag_RSTO'

        if (float(var3) == 2):
            v3 = 1
        else:
            v3 = 0

        # 'service_urh_i'
        if (float(var2) == 5):
            v4 = 1
        else:
            v4 = 0
        # 'flag_OTH'
        if (float(var3) == 0):
            v5 = 1
        else:
            v5 = 0

        # 'dst_host_serror_rate'
        v6 = float(var4)
        # 'diff_srv_rate',
        v7 = float(var5)
        # 'dst_host_same_src_port_rate'
        v8 = float(var6)
        # 'serror_rate'
        v9 = float(var7)
        'flag_RSTOS0'
        if (float(var3) == 3):
            v10 = 0
        else:
            v10 = 1
        # 'wrong_fragment'

        v11 = float(var8)
        # 'protocol_type_icmp'

        if (float(var9) == 0):
            v12 = 1
        else:
            v12 = 0
        # 'logged_in'

        v13 = float(var10)
        # 'srv_serror_rate',
        v14 = float(var11)
        # 'dst_host_same_srv_rate'
        v15 = float(var12)
        # 'flag_RSTR'
        if (float(var3) == 4):
            v16 = 1
        else:
            v16 = 0

        'is_host_login',

        v17 = float(var13)

        'is_guest_login'
        v18 = float(var14)
        'srv_diff_host_rate'
        v19 = float(var15)

        'service_eco_i'

        if (float(var2) == 0):
            v20 = 1
        else:
            v20 = 0

        'flag_REJ'

        if (float(var3) == 1):
            v21 = 1
        else:
            v21 = 0
        'flag_S0'

        if (float(var3) == 5):
            v22 = 1
        else:
            v22 = 0
        'service_red_i'

        if (float(var2) == 4):
            v23 = 1
        else:
            v23 = 0

        'dst_host_srv_count'

        v24 = float(var16)
        'count',
        v25 = float(var17)
        'same_srv_rate'
        v26 = float(var18)
        'service_pop_3'
        if (float(var2) == 3):
            v27 = 1
        else:
            v27 = 0

        'protocol_type_udp',

        if (var9 == 2):
            v28 = 1
        else:
            v28 = 0

        'dst_host_srv_diff_host_rate'
        v29 = float(var19)

        'flag_SF'
        if (float(var3) == 9):
            v30 = 1
        else:
            v30 = 0

        'srv_count',

        v31 = float(var20)
        'dst_host_diff_srv_rate',

        v32 = float(var21)
        'flag_S3'

        if (float(var3) == 8):
            v33 = 1
        else:
            v33 = 0

        'num_failed_logins'
        v34 = float(var22)
        'land',
        v35 = float(var23)
        'flag_SH',

        if (float(var3) == 10):
            v36 = 1
        else:
            v36 = 0
        'flag_S2',

        if (float(var3) == 7):
            v37 = 1
        else:
            v37 = 0

        'flag_S1'

        if (float(var3) == 6):
            v38 = 1
        else:
            v38 = 0

        'service_urp_i',

        if (float(var2) == 6):
            v39 = 1
        else:
            v39 = 0

        'protocol_type_tcp',

        if (var9 == 1):
            v40 = 1
        else:
            v40 = 0

        'service_ftp'

        if (float(var2) == 2):
            v41 = 1
        else:
            v41 = 0


        selected_algorithm = request.form.get('algorithm')

        # Implement the prediction logic based on the selected algorithm
        prediction_result = ''

        if selected_algorithm == 'fnn':
            # Use Feedforward Neural Network (FNN) for prediction
            # Implement FNN prediction logic here



            new_data = np.array([[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,
                                  v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,v41]])
            #print(new_data)

            newdf = pd.DataFrame(new_data, columns=inputvar)


            mindf=pd.read_csv('mindf.csv')
            maxdf=pd.read_csv('maxdf.csv')
            targetdf = (newdf - mindf) / (maxdf - mindf)
            new_data = targetdf.values

            # exporting the model
            import pickle
            #pickle.dump(model, open('attack_type_model_one_hot.pkl', 'wb'))
            model = pickle.load(open('dl_models/fnn/attack_type_model_one_hot.pkl', 'rb'))
            #print(new_data)
            prediction = model.predict(targetdf)
            import random
            l = [i for i in range(0, 40)]
            try:
                pr2 = np.argmax(prediction, axis=1)
                result =pr2[0]

                #print('prediction using fnn ',result)
            except:
                result = random.choice(l)





            prediction_result = col_list[result]  # Replace with the actual prediction result
            print('prediction using fnn is ',prediction_result)
        elif selected_algorithm == 'cnn':
            # Use Convolutional Neural Network (CNN) for prediction
            # Implement CNN prediction logic here
            # cnn algorithm coding for prediction
            import pickle
            model = pickle.load(open('dl_models/cnn/attack_type_model_one_hot_using_cnn.pkl', 'rb'))
            mindf = pd.read_csv('mindf.csv')

            maxdf = pd.read_csv('maxdf.csv')

            mindf

            maxdf

            new_data = np.array([[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,
                                  v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,v41]])



            newdf = pd.DataFrame(new_data, columns=inputvar)

            newdf

            targetdf = (newdf - mindf) / (maxdf - mindf)

            targetdf

            np.array(targetdf)[0]

            l = []

            for i in np.array(targetdf)[0]:
                l.append([i])

            l

            l = np.array([l])



            pr2 = model.predict(l)
            pr2 = np.argmax(pr2, axis=1)
            #print(pr2[0])

            prediction_result=col_list[pr2[0]]
            print('prediction using cnn is ', prediction_result)
        elif selected_algorithm == 'rnn':
            # Use recurrent Neural Network (CNN) for prediction
            # Implement RNN prediction logic here
            # rnn algorithm coding for prediction
            import pickle
            model = pickle.load(open('dl_models/rnn/attack_type_model_one_hot_using_rnn.pkl', 'rb'))
            mindf = pd.read_csv('mindf.csv')

            maxdf = pd.read_csv('maxdf.csv')
            l=[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21,
             v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41]
            for i in range(0,len(l)):
                if l[i]>40:
                    l[i]=40



            new_data = np.array(
                [l])

            newdf = pd.DataFrame(new_data, columns=inputvar)
            new_data = (newdf - mindf) / (maxdf - mindf)
            predictions = model.predict(new_data)
            predicted_classes = np.argmax(predictions, axis=1)
            result = col_list[predicted_classes[0]]
            prediction_result=result
            print('prediction using cnn is ', prediction_result)
        else:
            # Handle the case when the selected algorithm is not recognized
            return jsonify({'error': 'Invalid or unrecognized algorithm'})
        return prediction_result
    # Handle GET requests here if needed
    return 'GET request received'


@app.route('/predict_mlmodels', methods=['GET', 'POST'])
def predict_mlmodels():
    if request.method == 'POST':
        var1 = request.form.get('var1')
        var2 = request.form.get('var2')
        var3 = request.form.get('var3')
        var4 = request.form.get('var4')
        var5 = request.form.get('var5')
        var6 = request.form.get('var6')
        var7 = request.form.get('var7')
        var8 = request.form.get('var8')
        var9 = request.form.get('var9')
        var10 = request.form.get('var10')
        var11 = request.form.get('var11')
        var12 = request.form.get('var12')
        var13 = request.form.get('var13')
        var14 = request.form.get('var14')
        var15 = request.form.get('var15')
        var16 = request.form.get('var16')
        var17 = request.form.get('var17')
        var18 = request.form.get('var18')
        var19 = request.form.get('var19')
        var20 = request.form.get('var20')
        var21 = request.form.get('var21')
        var22 = request.form.get('var22')
        var23 = request.form.get('var23')
        # 'dst_host_srv_serror_rate'
        v1 = float(var1)
        # 'service_ecr_i'
        if (float(var2) == 1):
            v2 = 1
        else:
            v2 = 0
        # 'flag_RSTO'

        if (float(var3) == 2):
            v3 = 1
        else:
            v3 = 0

        # 'service_urh_i'
        if (float(var2) == 5):
            v4 = 1
        else:
            v4 = 0
        # 'flag_OTH'
        if (float(var3) == 0):
            v5 = 1
        else:
            v5 = 0

        # 'dst_host_serror_rate'
        v6 = float(var4)
        # 'diff_srv_rate',
        v7 = float(var5)
        # 'dst_host_same_src_port_rate'
        v8 = float(var6)
        # 'serror_rate'
        v9 = float(var7)
        'flag_RSTOS0'
        if (float(var3) == 3):
            v10 = 0
        else:
            v10 = 1
        # 'wrong_fragment'

        v11 = float(var8)
        # 'protocol_type_icmp'

        if (float(var9) == 0):
            v12 = 1
        else:
            v12 = 0
        # 'logged_in'

        v13 = float(var10)
        # 'srv_serror_rate',
        v14 = float(var11)
        # 'dst_host_same_srv_rate'
        v15 = float(var12)
        # 'flag_RSTR'
        if (float(var3) == 4):
            v16 = 1
        else:
            v16 = 0

        'is_host_login',

        v17 = float(var13)

        'is_guest_login'
        v18 = float(var14)
        'srv_diff_host_rate'
        v19 = float(var15)

        'service_eco_i'

        if (float(var2) == 0):
            v20 = 1
        else:
            v20 = 0

        'flag_REJ'

        if (float(var3) == 1):
            v21 = 1
        else:
            v21 = 0
        'flag_S0'

        if (float(var3) == 5):
            v22 = 1
        else:
            v22 = 0
        'service_red_i'

        if (float(var2) == 4):
            v23 = 1
        else:
            v23 = 0

        'dst_host_srv_count'

        v24 = float(var16)
        'count',
        v25 = float(var17)
        'same_srv_rate'
        v26 = float(var18)
        'service_pop_3'
        if (float(var2) == 3):
            v27 = 1
        else:
            v27 = 0

        'protocol_type_udp',

        if (var9 == 2):
            v28 = 1
        else:
            v28 = 0

        'dst_host_srv_diff_host_rate'
        v29 = float(var19)

        'flag_SF'
        if (float(var3) == 9):
            v30 = 1
        else:
            v30 = 0

        'srv_count',

        v31 = float(var20)
        'dst_host_diff_srv_rate',

        v32 = float(var21)
        'flag_S3'

        if (float(var3) == 8):
            v33 = 1
        else:
            v33 = 0

        'num_failed_logins'
        v34 = float(var22)
        'land',
        v35 = float(var23)
        'flag_SH',

        if (float(var3) == 10):
            v36 = 1
        else:
            v36 = 0
        'flag_S2',

        if (float(var3) == 7):
            v37 = 1
        else:
            v37 = 0

        'flag_S1'

        if (float(var3) == 6):
            v38 = 1
        else:
            v38 = 0

        'service_urp_i',

        if (float(var2) == 6):
            v39 = 1
        else:
            v39 = 0

        'protocol_type_tcp',

        if (var9 == 1):
            v40 = 1
        else:
            v40 = 0

        'service_ftp'

        if (float(var2) == 2):
            v41 = 1
        else:
            v41 = 0

        selected_algorithm = request.form.get('algorithm')

        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, classification_report
        import pickle
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        import pandas as pd
        data = pd.read_csv('processeddata.csv')
        data.head(2)
        l1 = ['dst_host_srv_serror_rate', 'service_ecr_i', 'flag_RSTO',
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
        x = data[l1]
        y = data['Class']
        # Sample categorical data
        categories = list(y)
        # Create a dictionary to map categories to numerical labels
        label_mapping = {category: label for label, category in enumerate(set(categories))}

        # Create a reverse mapping dictionary to map numerical labels back to categories
        reverse_mapping = {label: category for category, label in label_mapping.items()}

        # Create a lambda function for label encoding and reverse decoding
        label_encode_decode = lambda data, mapping: [mapping[category] for category in data]
        # Encode the original categories
        encoded_labels = label_encode_decode(categories, label_mapping)
        y_encoded = pd.Series(encoded_labels)
        y = y_encoded
        mindf = pd.read_csv('mindf.csv')
        maxdf = pd.read_csv('maxdf.csv')

        # Implement the prediction logic based on the selected algorithm
        prediction_result = ''

        new_data = np.array(
            [[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21,
              v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41]])

        newdf = pd.DataFrame(new_data, columns=inputvar)

        mindf = pd.read_csv('mindf.csv')
        maxdf = pd.read_csv('maxdf.csv')
        targetdf = (newdf - mindf) / (maxdf - mindf)













        if selected_algorithm == 'DecisionTreeClassifier':

            import pickle

            model = pickle.load(open('ml_models/DecisionTreeClassifier/DecisionTreeClassifier.pkl', 'rb'))
            #print(new_data)
            prediction = model.predict(targetdf)
            import random

            l = list(model.predict(targetdf))
            # Decode the encoded labels back to original categories
            decoded_categories = label_encode_decode(l, reverse_mapping)
            result = decoded_categories[0]
            prediction_result = result  # Replace with the actual prediction result
            print('prediction using DecisionTreeClassifier is ', prediction_result)
        elif selected_algorithm == 'KNeighborsClassifier':
            import pickle

            model = pickle.load(open('ml_models/KNeighborsClassifier/KNeighborsClassifier.pkl', 'rb'))
            #print(new_data)
            prediction = model.predict(targetdf)
            import random

            l = list(model.predict(targetdf))
            # Decode the encoded labels back to original categories
            decoded_categories = label_encode_decode(l, reverse_mapping)
            result = decoded_categories[0]
            prediction_result = result  # Replace with the actual prediction result
            print('prediction using KNeighborsClassifier is ', prediction_result)
        elif selected_algorithm == 'RandomForestClassifier':

            import pickle

            model = pickle.load(open('ml_models/RandomForestClassifier/RandomForestClassifier.pkl', 'rb'))
            #print(new_data)
            prediction = model.predict(targetdf)
            import random

            l = list(model.predict(targetdf))
            # Decode the encoded labels back to original categories
            decoded_categories = label_encode_decode(l, reverse_mapping)
            result = decoded_categories[0]
            prediction_result = result  # Replace with the actual prediction result
            print('prediction using RandomForestClassifier is ', prediction_result)

        elif selected_algorithm == 'SVC':
            import pickle

            model = pickle.load(open('ml_models/SVC/SVC.pkl', 'rb'))
            #print(new_data)
            prediction = model.predict(targetdf)
            import random

            l = list(model.predict(targetdf))
            # Decode the encoded labels back to original categories
            decoded_categories = label_encode_decode(l, reverse_mapping)
            result = decoded_categories[0]
            prediction_result = result  # Replace with the actual prediction result
            print('prediction using SVC is ', prediction_result)

        elif selected_algorithm == 'LogisticRegression':
            import pickle

            model = pickle.load(open('ml_models/LogisticRegression/LogisticRegression.pkl', 'rb'))
            #print(new_data)
            prediction = model.predict(targetdf)
            import random

            l = list(model.predict(targetdf))
            # Decode the encoded labels back to original categories
            decoded_categories = label_encode_decode(l, reverse_mapping)
            result = decoded_categories[0]
            prediction_result = result  # Replace with the actual prediction result
            print('prediction using LogisticRegression is ', prediction_result)


        elif selected_algorithm == 'MultinomialNB':
            import pickle

            model = pickle.load(open('ml_models/MultinomialNB/MultinomialNB.pkl', 'rb'))
            #print(new_data)
            prediction = model.predict(targetdf)
            import random

            l = list(model.predict(targetdf))
            # Decode the encoded labels back to original categories
            decoded_categories = label_encode_decode(l, reverse_mapping)
            result = decoded_categories[0]
            prediction_result = result  # Replace with the actual prediction result
            print('prediction using MultinomialNB is ', prediction_result)

        else:
            # Handle the case when the selected algorithm is not recognized
            return jsonify({'error': 'Invalid or unrecognized algorithm'})

        return prediction_result

            # Handle GET requests here if needed
    return 'GET request received'



































@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/attack_classification")
def attack_classification():
    return render_template("attack_classification.html")




























"""extra works"""
@app.route("/real_time_monitoring")
def real_time_monitoring():
    return render_template("real_time_monitoring.html")


from flask_socketio import SocketIO
#app = Flask(__name__)
socketio = SocketIO(app)
from flask_socketio import emit

# Define the event handler for 'my_event'
@socketio.on('my_event')
def handle_my_event(data):
    # Process incoming data and perform real-time monitoring tasks
    # You can send any data you want as the update
    update_data = {'message': 'Real-time update message'}
    socketio.emit('update', update_data, broadcast=True)  # Broadcast the update to all connected clients



import datetime


@socketio.on('my_event')
def handle_my_event(data):
    # 1. Receive and Parse Data (Example: Assuming 'data' is a dictionary)
    incoming_data = data.get('payload')

    # 2. Real-Time Monitoring Logic (Example: Dummy logic)
    if 'anomaly' in incoming_data:
        alert_message = f"Anomaly detected in {incoming_data['source']}"

    # 3. Update Data
    update_data = {
        'message': alert_message if 'alert_message' in locals() else 'No alerts',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 4. Emit Updates to Connected Clients
    socketio.emit('update', update_data, broadcast=True)


























































"""


from scapy.all import sniff
from scapy.all import IP, TCP, UDP




def process_packet(packet):
    if IP in packet:
        ip_packet = packet[IP]
        source_ip = ip_packet.src
        destination_ip = ip_packet.dst
        protocol = ip_packet.proto  # Protocol number (e.g., 6 for TCP, 17 for UDP)

        if protocol == 6 and TCP in ip_packet:
            tcp_packet = ip_packet[TCP]
            source_port = tcp_packet.sport
            destination_port = tcp_packet.dport

            # Extract payload (if needed)
            payload = str(tcp_packet.payload)

            # Perform further processing or analysis here
            print(f"Source IP: {source_ip}, Destination IP: {destination_ip}, Source Port: {source_port}, Destination Port: {destination_port}, Protocol: TCP")

        elif protocol == 17 and UDP in ip_packet:
            udp_packet = ip_packet[UDP]
            source_port = udp_packet.sport
            destination_port = udp_packet.dport

            # Extract payload (if needed)
            payload = str(udp_packet.payload)

            # Perform further processing or analysis here
            print(f"Source IP: {source_ip}, Destination IP: {destination_ip}, Source Port: {source_port}, Destination Port: {destination_port}, Protocol: UDP")

# Start capturing packets on the specified network interface (e.g., "eth0")
# Adjust the filter as needed to capture specific types of packets
sniff(iface="eth0", filter="ip", prn=process_packet)

"""
































"""from scapy.arch import get_if_list

# Get a list of available interfaces
interfaces = get_if_list()

for interface in interfaces:
    print(interface)





from scapy.all import sniff
from scapy.layers.inet import Ether, IP, TCP, ICMP,UDP
def process_packet(packet):
    if IP in packet:
        ip_packet = packet[IP]
        source_ip = ip_packet.src
        destination_ip = ip_packet.dst
        protocol = ip_packet.proto  # Protocol number (e.g., 6 for TCP, 17 for UDP)

        if protocol == 6 and TCP in ip_packet:
            tcp_packet = ip_packet[TCP]
            source_port = tcp_packet.sport
            destination_port = tcp_packet.dport

            # Extract payload (if needed)
            payload = str(tcp_packet.payload)

            # Perform further processing or analysis here
            print(f"Source IP: {source_ip}, Destination IP: {destination_ip}, Source Port: {source_port}, Destination Port: {destination_port}, Protocol: TCP")

        elif protocol == 17 and UDP in ip_packet:
            udp_packet = ip_packet[UDP]
            source_port = udp_packet.sport
            destination_port = udp_packet.dport

            # Extract payload (if needed)
            payload = str(udp_packet.payload)

            # Perform further processing or analysis here
            print(f"Source IP: {source_ip}, Destination IP: {destination_ip}, Source Port: {source_port}, Destination Port: {destination_port}, Protocol: UDP")

# Start capturing packets on the specified network interface (e.g., "eth0")
# Adjust the filter as needed to capture specific types of packets
sniff(iface="eth0", filter="ip", prn=process_packet)"""















if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.secret_key = 'your_secret_key_here'
    #app.run(debug=True,use_reloader=False)
    app.run(debug=True, use_reloader=False)
    """extra information"""

    socketio.run(app, debug=True)

















"""

<!--
        <a href="/analysis">ANALYSIS</a> <!-- Add this line for DLModels -->
        <a href="/attack_classification">ATTACK CLASSIFICATION</a> <!-- Add this line for DLModels -->
        <a href="/real_time_monitoring">REAL-TIME MONITORING</a> <!-- Add this line for DLModels -->
        <a href="/alerts_and_reports/">ALERTS AND REPORTS</a> <!-- Add this line for DLModels -->
        <a href="/configuration_settings">CONFIGURATION SETTINGS</a> <!-- Add this line for DLModels -->
        <a href="/user_management">USER MANAGEMENT</a> <!-- Add this line for DLModels -->
        <a href="/documentation_and_tutorials">DOCUMENTATION AND TUTORIALS</a> <!-- Add this line for DLModels -->
        <a href="/about_us/">ABOUT US</a> <!-- Add this line for DLModels -->
        <a href="/blog_news">BLOG_NEWS</a> <!-- Add this line for DLModels -->
        <a href="/data_privacy_and_security">DATA PRIVACY AND SECURITY</a> <!-- Add this line for DLModels -->
        <a href="/feedback_and_support">FEEDBACK AND SUPPORT</a> <!-- Add this line for DLModels -->
        <a href="/search_functionality/">SEARCH FUNCTIONALITY</a> <!-- Add this line for DLModels -->
        <a href="/social_media_integration">SOCIAL MEDIA INTEGRATION</a> <!-- Add this line for DLModels -->
        <a href="/data_visualizations/">DATA VISUALIZATIONS</a> <!-- Add this line for DLModels -->
        <a href="/downloadable_resources">DOWNLOADABLE RESOURCES</a> <!-- Add this line for DLModels -->
        -->
"""