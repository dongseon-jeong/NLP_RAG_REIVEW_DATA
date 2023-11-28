import pandas as pd
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import numpy as np
import operator
from langchain.vectorstores.utils import DistanceStrategy
import ast
import re


### 최종 아웃풋
class Search:
  
  def __init__(self,index_path,index_name):
    self.fdb = FAISS.load_local(index_path+index_name, OpenAIEmbeddings())
    self.embeddings = OpenAIEmbeddings()

  def search(self,k,threshold,query):
    embedding = self.embeddings.embed_query(query)
    vector = np.array([embedding], dtype=np.float32)

    if self.fdb._normalize_L2:
        FAISS.normalize_L2(vector)

    scores, indices = self.fdb.index.search(vector, k)

    docs = []
    for j, i in enumerate(indices[0]):
        if i == -1:
            continue
        _id = self.fdb.index_to_docstore_id[i]
        doc = self.fdb.docstore.search(_id)
        docs.append((doc, scores[0][j]))

    cmp = (
        operator.ge
        if self.fdb.distance_strategy
        in (DistanceStrategy.MAX_INNER_PRODUCT, DistanceStrategy.JACCARD)
        else operator.le
    )
    docs = [
        (doc, similarity)
        for doc, similarity in docs
        if cmp(similarity, threshold)
    ]

    samples_df = pd.DataFrame.from_dict(docs)
    samples_df = samples_df.rename(columns={0:"review",1:"scores"})
    samples_df.sort_values("scores", ascending=True, inplace=True)
    return samples_df
