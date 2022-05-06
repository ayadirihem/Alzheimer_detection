#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask,request,jsonify
import numpy as np
import pickle
# read our pickle file and label our logisticmodel as model
infile = open('model.pkl', 'rb')
model = pickle.load(infile, encoding='bytes')

print(model)

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    #(Gender,Age,EDUC,SES,MMSE,CDR,eTIV,nWBV,ASF)
    Gender = int(request.form.get('Gender'))
    print(Gender)
    Age = int(request.form.get('Age'))
    EDUC = float(request.form.get('EDUC'))
    SES = float(request.form.get('SES'))
    MMSE = float(request.form.get('MMSE'))
    CDR = float(request.form.get('CDR'))
    eTIV = float(request.form.get('eTIV'))
    nWBV = float(request.form.get('nWBV'))
    ASF = float(request.form.get('ASF'))
    input_query = tuple((Gender,Age,EDUC,SES,MMSE,CDR,eTIV,nWBV,ASF))
    # change the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_query)

    # reshape the numpy array as we are predicitng for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    if (prediction[0] == 0):
        result = 'The Person does not have a Alzheimer disease'
    elif prediction[0] == 1:
        result = 'This Person has Alzheimer disease'
    else:
        result = 'This Person start to have Alzheimer'
    return jsonify({'Result':str(result)})
if __name__ == '__main__':
    app.run()


# In[ ]:




