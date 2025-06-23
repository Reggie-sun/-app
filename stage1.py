import streamlit as st
from getpass import getpass
from dotenv import load_dotenv
load_dotenv() 
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
import os
api_key=os.getenv("OPENAI_API_KEY")
print("API Key:", os.getenv("OPENAI_API_KEY"))
from langchain_core.runnables import ConfigurableField
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from IPython.display import display, Markdown
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.tools import tool
prompt1 = ChatPromptTemplate.from_messages([
    ("system", 
        """请你讲一个主题是关于动作和冒险的故事，主角的性别是{Gender}这是一个适合孩子们看的童话故事，请用活泼、简单、充满想象力的语言来讲故事，不要太长（大约 300~500 字），要让孩子听了很开心也能学到知识。""",
    ),
    ("human", "{input}"),
])

prompt2 = ChatPromptTemplate.from_messages([
    ("system", 
        """请你讲一个主题是关于睡前故事的故事，主角的性别是{Gender}这是一个适合孩子们看的童话故事，请用活泼、简单、充满想象力的语言来讲故事，不要太长（大约 300~500 字），要让孩子听了很开心也能学到知识。"""
    ),
    ("human", "{input}"),
])

prompt3 = ChatPromptTemplate.from_messages([
    ("system", 
        """请你讲一个主题是关于充满希望和鼓舞的故事，主角的性别是{Gender}这是一个适合孩子们看的童话故事，请用活泼、简单、充满想象力的语言来讲故事，不要太长（大约 300~500 字），要让孩子听了很开心也能学到知识。"""
    ),
    ("human", "{input}"),
])

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.placeholder.markdown(self.text)        


from langchain_core.output_parsers import StrOutputParser
parser=StrOutputParser()

col1, col2 = st.columns([1, 4])

with col1:
    with st.sidebar:
        st.header('选择一个主题和主人公性别')
        st.selectbox('主题', ('动作与冒险', '睡前故事',"充满希望和鼓舞人心"))
        gender = st.selectbox('Gender', ('male', 'female'))

with col2:
    st.title('这是一个生成故事的app')

    input = st.text_area("请输入提示，以便得到你想要的故事，你可以命名主人公的名字，爱好等等,但是你也以选择不用提示，只选择左边的主题和性别直接生成故事")

    button=st.button("Please click here to generate a story.")

    placeholder = st.empty() 
    llm = ChatOpenAI(
            model="deepseek-chat",            
            api_key=api_key,  
            openai_api_base="https://api.deepseek.com",                 
            temperature=0.0,
            max_tokens=2048,
            streaming=True ,
            callbacks= [StreamHandler(placeholder)]
        ).configurable_fields(
            callbacks=ConfigurableField(
            id="callbacks",
            name="callbacks",
            description="A list of callbacks to use for streaming"
        ))
    chain1= prompt1 | llm |parser
    chain2= prompt2 | llm |parser
    chain3= prompt3 | llm |parser
    if button:
        placeholder = st.empty() 
        out= chain1.invoke({
                "input":input,
                 "Gender":gender
                 } )
        
        







    




