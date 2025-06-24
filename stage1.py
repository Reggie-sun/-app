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
        """Please tell a story with the theme of {theme} and the protagonist's gender being {Gender}. 
        This should be a fairy tale suitable for children, told in lively, simple and imaginative language. 
        The story must be divided into exactly {pages} pages.Maximum of 10 pages generated. Each page should be clearly numbered and labeled,
          and the story should end at the {pages}th page. Make sure the length of the story is evenly distributed across the pages
          .Users can change the storyline according to their input.so that children can enjoy it while also learning something.
          And the page numbers for each page should also be displayed.You need to do it like you did for each page, because I will also insert images later. 
          The stories you create should have different contents and depths for children of different ages,this is the{age}.
          The type of the character can also be selected,this is the{character}
          The character's personality traits can also be selected,this is the {personality_traits}
          The hobbies and can also be selected.this is the{hobbies}
          Background story can also be selected,this is the {background}
          The theme for educational content can also be chosen,this is the {moral}
          Place the app's title and the user's input on the first page, and all the subsequent content should be articles that can be scrolled through.
          Ultimately, the user's input should be given priority. If the user's input conflicts with what you have set, the user's input should take precedence.Ultimately, the user's input should be the priority. If the user's input conflicts with what you have set, the user's input should take precedence."""
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
    st.header('Select the features you wish to input.')
    theme = st.selectbox('theme', ('Action and Adventure', 
                                   "bedtime story",
 "Full of hope and inspiring"))
    gender = st.selectbox('Gender', ('male', 'female'), key="gender_selectbox")
    pages=st.selectbox('pages',("1",'2','3','4','5','6','7','8','9','10'),key="gender_page")
    age=st.selectbox("age",("From 1 to 3 years old","4 to 6 years old","7 to 9 years old","10 to 13 years old"))
    character=st.selectbox('character_type',("Human", "Animal", "Robot", "Alien", "Magical creature"))
    personality_traits=st.selectbox('Personality traits',("Brave", "Curious", "Kind", "Funny", "Shy", "Clever"))
    hobbies=st.selectbox('hobbies',("Drawing", "Flying", "Exploring", "Reading", "Cooking", "Solving puzzles"))
    background=st.selectbox('background',("Fantasy Kingdom", "Deep Space", "Jungle", "Underwater World", "Toyland", "City at night"))
    moral=st.selectbox('moral',("Friendship", "Honesty", "Bravery", "Creativity", "Teamwork", "Self-love"))

st.title('This is an app that generates stories.')

input = st.text_area(
    "Please provide some hints so that you can get the fairy tale you want. Of course, you don't have to choose the hints; you can simply select them from the columns on the left sidebar to generate the story.")

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
            "theme": theme,
            "pages":pages,
            "age":age,
            "character":character,
            "personality_traits":personality_traits,
            "hobbies":hobbies,
            "background":background,
            "moral":moral
        })
