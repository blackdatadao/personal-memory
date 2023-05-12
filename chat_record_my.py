
import requests
import json
import streamlit as st

url='http://42.192.17.155/chat_record_my'
response = requests.get(url)
assert response.status_code==200
data=response.json()
for d in data:
    with st.container():
            st.write("你: "+d['ask'])
            st.write("Timothy: "+d['answer'])
            st.write('---')
