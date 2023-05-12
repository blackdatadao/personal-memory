
import requests
import json
import streamlit as st

url='http://42.192.17.155/chat_record_liu'
response = requests.get(url)
assert response.status_code==200
data=response.json()
for d in data:
    with st.container():
            st.markdown(d['ask'])
            st.markdown(d['answer'])
            st.write('---')
