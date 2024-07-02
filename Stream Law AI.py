import streamlit as st


from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pyttsx3
st.title("AI Lawyer Consultent")
engine = pyttsx3.init()
#engine.setProperty('voice', voices[0].id)
data=pd.read_csv('AI law data.csv')

vector=CountVectorizer()
a=vector.fit_transform(data['Cases'])
dtree=DecisionTreeClassifier()
dtree=dtree.fit(a,data['Solutions'])
st.subheader("This is for informational purposes only. For legal advice, consult a professional")
mes = st.text_input("Enter Your Query")
check = st.button("ASK")
if check:
    st.write(mes)
    b = vector.transform([mes])
    st.success(dtree.predict(b)[0])
    engine.say(dtree.predict(b)[0])
    engine.runAndWait()
