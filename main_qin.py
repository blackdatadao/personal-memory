# -*- coding: utf-8 -*-
# #.\env\Scripts\activate


import pandas as pd


import streamlit as st
import openai
import os,requests,json
import streamlit.components.v1 as components

import ast
from scipy import spatial
import tiktoken


EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 1600
AI_CLONE = "QIN"
name="秦朔"

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
    """Split a string in two, on a delimiter, trying to balance tokens on each side."""
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # no delimiter found
    elif len(chunks) == 2:
        return chunks  # no need to search for halfway point
    else:
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]

def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = True,
) -> str:
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_string

def split_strings_from_subsection(
    subsection: tuple[list[str], str],
    max_tokens: int = 1000,
    model: str = GPT_MODEL,
    max_recursion: int = 5,
) -> list[str]:
    """
    Split a subsection into a list of subsections, each with no more than max_tokens.
    Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
    """
    titles, text = subsection
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)
    # if length is fine, return string
    if num_tokens_in_string <= max_tokens:
        return [string]
    # if recursion hasn't found a split after X iterations, just truncate
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    # otherwise, split in half and recurse
    else:
        titles, text = subsection
        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(text, delimiter=delimiter)
            if left == "" or right == "":
                # if either half is empty, retry with a more fine-grained delimiter
                continue
            else:
                # recurse on each half
                results = []
                for half in [left, right]:
                    half_subsection = (titles, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
    # otherwise no split was found, so just truncate (should be very rare)
    return [truncated_string(string, model=model, max_tokens=max_tokens)]

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    
    introduction = '你的角色是秦朔，你是秦朔，秦朔是你。只在下面的文章中查找后面问题的答案。\
    下面这些文章都是秦朔写的。\
    如果在文章中找不到精确的答案，请回答我不知道。\
    如果问题里询问除秦朔之外的其他人的观点，而在文章中找到的观点不是这个人的，请回答我不知道。\
    如果问题里出现了可能是人名的情况，当你不确定它是不是人名的时候，请先反问这是一个人的名字吗\
    如果在文章中仅找到问题中的个别名词，却没有找到这些名词是什么、没有找到这个名词和其他名词有什么关联的信息，也没有这个名词发生了什么的信息，请回答我掌握的信息不够。\
    如果问题问题中涉及评论现任国家领导人和政党，请回答我不评论政治'
    question = f"\n\n问题: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\n文章:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    print(message + question)
    return message + question

def ask(
    query: str,
    df: pd.DataFrame,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "你是秦朔，用第一人称、秦朔的口吻在详细、友善、有条理的回答问题。"},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

def query_direct(text,question):
    query = f"""Use the below article to answer the subsequent question. If the answer cannot be found, write "I don't know."

    Article:
    \"\"\"
    {text}
    \"\"\"

    Question:{question}"""

    response = openai.ChatCompletion.create(
        messages=[
            {'role': 'system', 'content': 'You answer questions.Answer should be in Chinese'},
            {'role': 'user', 'content': query},
        ],
        model=GPT_MODEL,
        temperature=0,
    )
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']

def read_text(path):
    text_list=[]
    for file in os.listdir(path):
        with open(os.path.join(path,file),'r',encoding='utf-8') as f:
            
            text=f.read()
            print(file,' read completed')
        title=file.split('.')[0]
        text_list.append(([title],text))
    return text_list

def read_text_get_embedding(read_path,save_path):
    wikipedia_strings = []
    wikipedia_sections=read_text(read_path)
    for section in wikipedia_sections:
        wikipedia_strings.extend(split_strings_from_subsection(section, max_tokens=MAX_TOKENS))
    print(f"{len(wikipedia_sections)} Wikipedia sections split into {len(wikipedia_strings)} strings.")
    EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
    BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

    embeddings = []
    for batch_start in range(0, len(wikipedia_strings), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = wikipedia_strings[batch_start:batch_end]
        print(f"Batch {batch_start} to {batch_end-1}")
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
        for i, be in enumerate(response["data"]):
            assert i == be["index"]  # double check embeddings are in same order as input
        batch_embeddings = [e["embedding"] for e in response["data"]]
        embeddings.extend(batch_embeddings)

    df = pd.DataFrame({"text": wikipedia_strings, "embedding": embeddings})
    df.to_csv(save_path, index=False)
    return df

#create a function to add df to a existing df from a csv file
def add_txt_to_embedding(txt_path,csv_path,save_path):
    df1=pd.read_csv(csv_path)
    df2=read_text_get_embedding(txt_path,save_path)
    df=pd.concat([df1,df2],ignore_index=True)
    df.to_csv(save_path,index=False)
    return df


def send_data_to_server(question,answer):
    url='http://42.192.17.155/chat_record_my'
    data = {
    "question": question,
    "answer": answer,
    # ... and so on
    }
    json_data = bytes(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    # Send the JSON data to the server
    response = requests.post(url, data=json_data)
    # Check if the POST request was successful
    if response.status_code == 200:
        print("Data successfully sent to server.")
    else:
        print("Failed to send data to server.")

st.secrets.load_if_toml_exists()
openai.api_key = st.secrets["openai_api_key"]
# openai.organization = st.secrets["openai_organization"]
assert openai.api_key is not None, "OpenAI API key not found"
os.environ["OPENAI_API_KEY"] = openai.api_key
# os.environ["OPENAI_ORGANIZATION"] = openai.organization


st.title(f"{AI_CLONE}'s clone")

df = pd.read_csv("example_data/qin.csv")
df['embedding'] = df['embedding'].apply(ast.literal_eval)

history = []

def chat(user_input: str) -> str:
    # only take last 3 messages (in practice, we should ensure it doesn't exceed the max length of the prompt)
    history_prefix = "\n---\n".join(history[-3:])
    response = ask(f"{history_prefix}\n---\n你: {user_input}\n{AI_CLONE}:",df)
    history.append(f"你: {user_input}\n{AI_CLONE}: {response}")
    return response

# input box
user_input = st.text_input("你", "你是谁？")
# button
if st.button("Send"):
    # display user input
    st.write("You: " + user_input)
    # display clone response
    response = chat(user_input)
    send_data_to_server(user_input,response)
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
    **⚠️ Note:** 测试中...
    """
)


# reference to gpt-index
st.markdown(
    """
    仅做产品demo，请勿公开发布在朋友圈、群。
    """
)
