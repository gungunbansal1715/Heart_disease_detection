import streamlit as st
import pandas as pd
import pickle
import base64

def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV File</a>'
    return href

st.title("HEART DISEASE PREDICTION APP")
tab1 , tab2 , tab3 = st.tabs(["Home","Prediction","About Us"])

with tab1:
    age = st.number_input("Enter Age", min_value=1, max_value=120, value=25)
    sex = st.selectbox('Sex(0-female, 1-male)', [0, 1])
    cp = st.selectbox('Chest Pain Type(1-typical angina, 2-atypical angina, 3-non-anginal pain, 4-asymptomatic)', [1,2,3,4])
    restingbp = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Cholestoral in mg/dl", min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)', [0, 1])
    restecg = st.selectbox('Resting Electrocardiographic Results(0-normal, 1-having ST-T wave abnormality, 2-left ventricular hypertrophy)', [0,1,2])
    maxhr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exerciseangina = st.selectbox('Exercise Induced Angina (1 = yes; 0 = no)', [0, 1])
    oldpeak = st.number_input("Oldpeak (ST depression induced by exercise relative to rest)", min_value=0.0, max_value=10.0, value=1.0)
    st_slope = st.selectbox('Slope of the peak exercise ST segment (1-upsloping, 2-flat, 3-downsloping)', [1,2,3])
    
    input_data = pd.DataFrame({
        'age': [age],
        'sex' : [sex],
        'chest pain type': [cp],
        'resting bp s': [restingbp],
        'cholesterol': [chol],
        'fasting blood sugar': [fbs],
        'resting ecg': [restecg],
        'max heart rate': [maxhr],
        'exercise angina': [exerciseangina],
        'oldpeak': [oldpeak],
        'ST slope': [st_slope],
    })
    algonames = ['Logistic Regression', 'Decision Tree', 'Random Forest',  'SVM']
    modelnames = ['LogisticR.pkl' , 'DecisionTree.pkl', 'RandomForest.pkl', 'SVM.pkl']

    predictions = []
    def predict_heart_disease(data):
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions
    
    if st.button("Predict"):
        st.subheader('Results...........')
        st.markdown('---')

        result = predict_heart_disease(input_data)

        for i in range(len(predictions)):
            st.header(algonames[i])
            if result[i] == 0:
                st.write("No Heart Disease")
            else:
                st.write("Heart Disease Detected")
            st.markdown('----------')

with tab2:
    st.title("Here we will predict disease in bulk")
    st.subheader("Before uploading the csv file make sure to follow the following instrucutions: ")
    st.info("""
        1. No NaN values are allowed.
        2. The csv file should contain the following columns in the exact order:
            age , sex , chest pain type , resting bp s , cholesterol , fasting blood sugar , resting ecg , max heart rate , exercise angina , oldpeak , ST slope
        3. check the spelling of the feature names.
            """)
    
    uploaded_file = st.file_uploader("Choose a file" , type = ["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        model = pickle.load(open('LogisticR.pkl', 'rb'))

        expected_columns = ['age' , 'sex' , 'chest pain type' , 'resting bp s' , 'cholesterol' , 'fasting blood sugar' , 'resting ecg' , 'max heart rate' , 'exercise angina' , 'oldpeak' , 'ST slope']

        if set(expected_columns).issubset(data.columns):
            predictions = model.predict(data[expected_columns])
            data['predictions'] = predictions
            data.to_csv('PREDICTION.csv', index=False)

            st.subheader("predictions")
            st.write(data)


            st.markdown(get_binary_file_downloader_html(data), unsafe_allow_html=True)
        else:
            st.warning("Please make sure the uploaded csv file has the correct columns.")
    else:
        st.info("Upload a csv file to get predictions. ")

with tab3:
    st.title("About Us")
    st.info("""
        This app is developed by a team of data science enthusiasts to help people predict the likelihood of heart disease based on various health parameters.
        The app uses machine learning algorithms to analyze the input data and provide predictions.
        We hope this app helps you in your health journey
        This app is developed by:
            Gungun Bansal
            contact: bansalgungun203@gmail.com
            A data science enthusiast 
            This project is developed as a part of machine learning course on Unified Mentor. 
            """)
    import plotly.express as px
    data = {'Logistic Regression': 0.85,
            'Decision Tree': 0.82,
            'Random Forest': 0.86,
            'SVM': 0.84}
    models = list(data.keys())
    accuracy = list(data.values())
    df = pd.DataFrame(list(zip(models, accuracy)), columns=['model', 'accuracy'])
    fig = px.pie(df, values='accuracy', names='model', title='Model Accuracy Comparison')
    st.plotly_chart(fig)
    fig1 = px.bar(df, x='model', y='accuracy', title='Model Accuracy Comparison', text='accuracy')
    st.plotly_chart(fig1)