import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse',False)

st.title('Breast Cancer Prediction')
st.subheader('Does the patient have breast cancer')
df = pd.read_csv('bcancer.csv')

if st.sidebar.checkbox('View Data',False):
    st.write(df)
if st.sidebar.checkbox('View Distribution',False):
    df.hist()
    plt.tight_layout()
    st.pyplot()
    

model = open('rfc.pickle','rb')
clf = pickle.load(model)
model.close()

radius_mean = st.slider('radius_mean',40,200,40)
texture_mean = st.slider('texture_mean',40,200,40) 
perimeter_mean = st.slider('perimeter_mean',40,200,40) 
area_mean = st.slider('area_mean',40,200,40) 
smoothness_mean = st.slider('smoothness_mean',40,200,40) 
compactness_mean = st.slider('compactness_mean',40,200,40) 
concavity_mean = st.slider('concavity_mean',40,200,40) 
concave_points_mean = st.slider('concave points_mean',40,200,40) 
symmetry_mean = st.slider('symmetry_mean',40,200,40) 
fractal_dimension_mean = st.slider('fractal_dimension_mean',40,200,40) 
radius_se = st.slider('radius_se',40,200,40) 
texture_se = st.slider('texture_se',40,200,40) 
perimeter_se = st.slider('perimeter_se',40,200,40) 
area_se = st.slider('area_se',40,200,40) 
smoothness_se = st.slider('smoothness_se',40,200,40) 
compactness_se = st.slider('compactness_se',40,200,40) 
concavity_se = st.slider('concavity_se',40,200,40) 
concave_points_se = st.slider('concave points_se',40,200,40) 
symmetry_se = st.slider('symmetry_se',40,200,40) 
fractal_dimension_se = st.slider('fractal_dimension_se',40,200,40) 
radius_worst = st.slider('radius_worst',40,200,40) 
texture_worst = st.slider('texture_worst',40,200,40) 
perimeter_worst = st.slider('perimeter_worst',40,200,40) 
area_worst = st.slider('area_worst',40,200,40) 
smoothness_worst = st.slider('smoothness_worst',40,200,40) 
compactness_worst = st.slider('compactness_worst',40,200,40) 
concavity_worst = st.slider('concavity_worst',40,200,40) 
concave_points_worst = st.slider('concave points_worst',40,200,40) 
symmetry_worst = st.slider('symmetry_worst',40,200,40) 
fractal_dimension_worst = st.slider('fractal_dimension_worst',40,200,40) 

input_data = [[radius_mean,texture_mean,perimeter_mean,area_mean,
              smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,
              symmetry_mean,fractal_dimension_mean,radius_se,texture_se,
              perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,
              concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,
              texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,
              concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]]

prediction = clf.predict(input_data)[0]
if st.button('Predict'):
    if prediction == 1:
        st.subheader('Malignant')
    else:
        st.subheader('Benign')
