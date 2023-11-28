from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import re
import pandas as pd
from langchain.vectorstores import FAISS

### 문장 생성
class Create_Sentence:

  def __init__(self ,index_path, index_name ,template, human_template):
    self.fdb = FAISS.load_local(index_path+index_name, OpenAIEmbeddings())
    self.chat = ChatOpenAI(temperature=0)
    self.chat_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(template), 
                                                         HumanMessagePromptTemplate.from_template(human_template)])
    self.memory = VectorStoreRetrieverMemory(retriever=self.fdb.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": .5,"k": 10}))
    self.texts = []

  def create_rag(self,text):
      self.texts = LLMChain(llm=self.chat, prompt=self.chat_prompt, verbose=False, memory=self.memory).run(text=text)
      return self.texts

  def create(self,text):
      self.texts = LLMChain(llm=self.chat, prompt=self.chat_prompt, verbose=False).run(text=text)
      return self.texts

