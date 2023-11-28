from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain.agents.agent_toolkits import VectorStoreInfo
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
import json
from pathlib import Path
from pprint import pprint
import os
import pandas as pd




class Review_Embedding:

  def __init__(self, in_path):
      self.x = 0
      self.file_list = []
      self.dataset = pd.read_csv(in_path + "review_tags.csv")
      self.list = []
      self.length = 0
      self.file_list = []
      self.file_list_py = []
      self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
      self.embeddings = OpenAIEmbeddings()
      self.vectorstore_info = []


  def len(self):
      self.length = round(len(self.dataset) / 100000) + 1
      return self.length

  def len_list(self):
      self.list = [i * 100000 for i in range(1, self.length + 1)]
      return self.list

  def create_dataset(self,out_path):
    for j, val in enumerate(self.list):
      globals()["dataset{}".format(j)] = self.dataset.iloc[self.x:val,]
      self.x = val + 1
      self.file_list.append(f'dataset{j}')
    # 데이터셋 csv파일로 저장
    for j, file in enumerate(self.file_list):
      globals()["dataset{}".format(j)].to_csv(out_path + f'./dataset{j}.csv',encoding = "utf-8")
    # 데이터셋 변수 제거
    for j, file in enumerate(self.file_list):
      del globals()["dataset{}".format(j)]

  def create_index(self,file_path):
    self.file_list = os.listdir(file_path)
    self.file_list_py = [file for file in self.file_list if file.endswith(".csv")]
      
    for i,file in enumerate(self.file_list_py):
      index = VectorstoreIndexCreator(
          vectorstore_cls=FAISS,
          embedding=self.embeddings,
          text_splitter=self.text_splitter,
          ).from_loaders([CSVLoader( 
             file_path=file_path+file,  
               source_column="message", 
               encoding = "utf-8",
               csv_args={
                    'delimiter': ',',
                    'quotechar': '"'
                    ,'fieldnames': ['Unnamed: 0.1', 'Unnamed: 0', 'brand_code', 'id', 'sentiment',
                                    'review_id', 'review_tag_type_id', 'keywords', 'message', 'product_id',
                                    'product_name', 'created_at']})])
      index.vectorstore.save_local(file_path+f"faiss{i}")

  # 10만개 이상 있을 경우
  def merge_index(self, file_path):
    for i in enumerate(self.file_list_py):
      self.globals()["fdb{}".format(i)] = FAISS.load_local(file_path+f"faiss{i}", self.embeddings)
      self.vectorstore_info = VectorStoreInfo(
        name="i",
        description="i",
        vectorstore=globals()["fdb{}".format(i)]
        )
    for i in enumerate(self.file_list_py):
      self.fdb0.merge_from(file_path+globals()["fdb{}".format(i)])
      del globals()["fdb{}".format(i)]

    self.fdb0.save_local(self.fdb0, folder_path = file_path+f"faiss_total")




# class Review_Embedding_HU:

#   def __init__(self, in_path, model):
#       self.x = 0
#       self.file_list = []
#       self.dataset = pd.read_csv(in_path + "review_tags.csv")
#       self.list = []
#       self.length = 0
#       self.file_list = []
#       self.file_list_py = []
#       self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
#       self.embeddings = HuggingFaceEmbeddings(model_name=model,model_kwargs = {'device': 'cpu'},encode_kwargs = {'normalize_embeddings': False})
#       self.vectorstore_info = []


#   def len(self):
#       self.length = round(len(self.dataset) / 100000) + 1
#       return self.length

#   def len_list(self):
#       self.list = [i * 100000 for i in range(1, self.length + 1)]
#       return self.list

#   def create_dataset(self,out_path):
#     for j, val in enumerate(self.list):
#       globals()["dataset{}".format(j)] = self.dataset.iloc[self.x:val,]
#       self.x = val + 1
#       self.file_list.append(f'dataset{j}')
#     # 데이터셋 csv파일로 저장
#     for j, file in enumerate(self.file_list):
#       globals()["dataset{}".format(j)].to_csv(out_path + f'./dataset{j}.csv',encoding = "utf-8")
#     # 데이터셋 변수 제거
#     for j, file in enumerate(self.file_list):
#       del globals()["dataset{}".format(j)]

#   def create_index(self,file_path):
#     self.file_list = os.listdir(file_path)
#     self.file_list_py = [file for file in self.file_list if file.endswith(".csv")]
      
#     for i,file in enumerate(self.file_list_py):
#       index = VectorstoreIndexCreator(
#           vectorstore_cls=FAISS,
#           embedding=self.embeddings,
#           text_splitter=self.text_splitter,
#           ).from_loaders([CSVLoader( 
#              file_path=file_path+file,  
#                source_column="message", 
#                encoding = "utf-8",
#                csv_args={
#                     'delimiter': ',',
#                     'quotechar': '"'
#                     ,'fieldnames': ['Unnamed: 0.1', 'Unnamed: 0', 'brand_code', 'id', 'sentiment',
#                                     'review_id', 'review_tag_type_id', 'keywords', 'message', 'product_id',
#                                     'product_name', 'created_at']})])
#       index.vectorstore.save_local(file_path+f"faiss{i}")

#   def merge_index(self, file_path):
#     for i in enumerate(self.file_list_py):
#       self.globals()["fdb{}".format(i)] = FAISS.load_local(file_path+f"faiss{i}", self.embeddings)
#       self.vectorstore_info = VectorStoreInfo(
#         name="i",
#         description="i",
#         vectorstore=globals()["fdb{}".format(i)]
#         )
#     for i in enumerate(self.file_list_py):
#       self.fdb0.merge_from(file_path+globals()["fdb{}".format(i)])
#       del globals()["fdb{}".format(i)]

#     self.fdb0.save_local(self.fdb0, folder_path = file_path+f"faiss_total")
