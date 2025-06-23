from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.tools import tool
from langchain_core.runnables.base import RunnableSerializable
from IPython.display import display, Markdown
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField
import os
from langchain_community.chat_models import ChatZhipuAI
from langchain_openai import ChatOpenAI
import streamlit as st
from getpass import getpass
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("API Key:", os.getenv("OPENAI_API_KEY"))
prompt = ChatPromptTemplate.from_messages([
    ("system",
        """Please tell a story with the theme of {theme} and the protagonist's gender being {Gender}. This should be a fairy tale suitable for children, told in lively, simple and imaginative language. It should not be too long (about 300 to 500 words), so that children can enjoy it while also learning something.""",
     ),
    ("human", "{input}")
])


class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.placeholder.markdown(self.text)


parser = StrOutputParser()


# 在columns里，用selectbox会导致streamlit内部检索id出错，所以要用唯一的 key

with st.sidebar:
    st.header('Choose a theme and the gender of the protagonist.')
    theme = st.selectbox('theme', ('Action and Adventure', "bedtime story
", "Full of hope and inspiring"))
    gender = st.selectbox('Gender', ('male', 'female'), key="gender_selectbox")

st.title('This is an app that generates stories.')

input = st.text_area(
    "Please provide a prompt so that we can generate the story you desire. You can name the protagonist's name, hobbies, etc. However, you can also choose not to provide a prompt and simply select the topic and gender to generate the story directly.")

button = st.button("Please click here to generate a story.")

placeholder = st.empty()
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    openai_api_base="https://api.deepseek.com",
    temperature=0.0,
    max_tokens=2048,
    streaming=True,
    callbacks=[StreamHandler(placeholder)]
).configurable_fields(
    callbacks=ConfigurableField(
        id="callbacks",
        name="callbacks",
        description="A list of callbacks to use for streaming"
    ))
chain = prompt | llm | parser

if button:
    placeholder = st.empty()
    out = chain.invoke({
            "input": input,
            "Gender": gender,
            "theme": theme
        })
