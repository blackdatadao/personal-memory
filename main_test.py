# -*- coding: utf-8 -*-
# #.\env\Scripts\activate

from langchain.llms import OpenAI

from langchain.document_loaders import TextLoader
from llama_index import QuestionAnswerPrompt, GPTVectorStoreIndex, SimpleDirectoryReader,GPTTreeIndex
from llama_index import StorageContext, load_index_from_storage

import streamlit as st
import openai
import os
import streamlit.components.v1 as components


def qury_from_storage_index(question):
    
    QA_PROMPT_TMPL = (
    "We have provided context information below\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str},answer should be in Chinese beginning with 甄科学认为.\n"
    )
    QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

    query_engine = index.as_query_engine(text_qa_template=QA_PROMPT)
#     query_engine = index.as_query_engine()

    response = query_engine.query(question).response
    return response 


st.secrets.load_if_toml_exists()
openai.api_key = st.secrets["openai_api_key"]
# openai.organization = st.secrets["openai_organization"]
assert openai.api_key is not None, "OpenAI API key not found"
os.environ["OPENAI_API_KEY"] = openai.api_key
# os.environ["OPENAI_ORGANIZATION"] = openai.organization

AI_CLONE = "甄科学"
st.title(f"{AI_CLONE}'s clone")

storage_context = StorageContext.from_defaults(persist_dir="./storage2")
index = load_index_from_storage(storage_context)
history = []

def chat(user_input: str) -> str:
    # only take last 3 messages (in practice, we should ensure it doesn't exceed the max length of the prompt)
    history_prefix = "\n---\n".join(history[-3:])
    response = qury_from_storage_index(f"{history_prefix}\n---\nHuman: {user_input}\n{AI_CLONE}:")
    history.append(f"Human: {user_input}\n{AI_CLONE}: {response}")
    return response

# input box
user_input = st.text_input("你", "AI会有哪些应用？")
# button
if st.button("Send"):
    # display user input
    st.write("You: " + user_input)
    # display clone response
    response = chat(user_input)
    st.write(f"{AI_CLONE}: " + response)

components.html(
    """
<script>
const doc = window.parent.document;
buttons = Array.from(doc.querySelectorAll('button[kind=primary]'));
const send = buttons.find(el => el.innerText === 'Send');
doc.addEventListener('keydown', function(e) {
    switch (e.keyCode) {
        case 13:
            send.click();
            break;
    }
});
</script>
""",
    height=0,
    width=0,
)

# add a description on how it differs from ChatGPT
# to be used as a personality for the AI

# add a warning / reminder that there a small conversation history
# of the past 3 messages that is used for the conversation
# something a nontech person understand
st.markdown(
    """
    **⚠️ Note:** The AI remembers only the past 3 messages. If you want to start a new conversation, refresh the page.
    """
)


# reference to gpt-index
st.markdown(
    """
    甄科学AI实验室出品
    """
)


