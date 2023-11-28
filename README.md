# RAG

## langchain
파이프라인  
chat  
agent  
memory  
생성 테스트  

## 임베딩 모델 선택
**허깅페이스모델  
klue/roberta-small : [[klue/roberta-small](https://huggingface.co/klue/roberta-small)]  
kt믿음 : [[KT-AI/midm-bitext-S-7B-inst-v1](https://huggingface.co/KT-AI/midm-bitext-S-7B-inst-v1)]  

**오픈ai 모델  
text-embedding-ada-002 : [링크](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)

## 리트리버
문서 추출  
텍스트 스플리터    

## 벡터 DB
파인콘  
faiss  
elestic search  

## 인덱스 생성
저장  

## RAG 활용
기존 konlpy mecab 토크나이저를 활용한 워드클라우드를 개선하기 위해 아래와 같은 방법을 시도하였습니다.  
- 명사만 형태소 추출한다. 대시보드 워드클라우드에도 명사만 노출  
- 주요명사에 인접한 ‘동사’형용사’ 단어는 chatGPT로 생성  
- 명사+동사 문장을 이용하여 임베딩 유사 리뷰 검색 집계  

하이라이트에 키워드가 중복되는 부분을 개선하기 위해 빈도수가 높은 토큰을 키워드 분류 및 직관적인 메세지 전달을 위한 문장 생성하고  
생성된 문장을 집계하기 위해 실시간 유사도 검색하여 집계하는 방법을 구현했습니다.  

![ex_screenshot](./img/process.png)  

1. 모든 리뷰를 임베딩모델에 넣어 벡터로 변환 > 문장 생성 시 랭체인 메모리 활용  
2. 빈도수 높은 명사 추출 후 메모리 활용하여 문장 생성  
3. 생성된 문장도 벡터로 변환 및 키워드 분류  
4. 마지막으로 대시보드에서 선택한 생성 문장에 유사한 리뷰 검색하여 집계  


### 전처리 테이블 생성
빈도수 높은 명사를 추출하는 과정입니다.
```python
from preprocessing import review_tag, re90

review_tag(in_path,out_path,date)
re90(in_path,out_path)
```

### 임베딩 생성
오픈ai 임베딩 모델의 api 결과를 받아 faiss 인덱스를 저장하는 과정입니다.
```python
from review_embedding import Review_Embedding

review_embedding = Review_Embedding(out_path)
review_embedding.create_dataset(out_path)
review_embedding.create_index(index_path)
```

### 문장 생성
랭체인 메모리 활용 코드입니다.  
```python
from sentence_create import Create_Sentence

nn = pd.read_csv(out_path+"re90.csv")
text = nn['token'][0]
template="""
1.당신은 명사 다음에 올 동사나 형용사를 구하는 ai입니다.

2.최종 대표적인 동사, 형용사 최대 4개를 추출합니다.

3.형용사나 동사는 '다'로 끝나는 형태로 출력합니다.
예시
짧은, 짧아서 > 짧다
길어서, 긴 > 길다

4.답변은 단답형으로 합니다.
예시
입력: 길이
출력: 짧다,길다
"""
human_template="다음 명사 다음 단어를 구해줘, {text}"

create_sentence = Create_Sentence(index_path,"faiss0",template,human_template)
create_sentence.create_rag(text)
```

### 키워드 분류
단순 원샷 러닝 생성입니다.  
```python
from sentence_create import Create_Sentence

text = "예뻐요"
template="""
1.당신은 키워드를 분류하는 ai입니다.

2.키워드 14개이고, 다음과 같습니다.
'가격','기능성','길이','디자인','라인(핏)','마감처리','배송','사이즈','색상','소재','스타일','신축성','착용감','품질'

3.가장 유사한 의미를 나타내는 키워드로 분류합니다.
예시
몸에 맞다 > 라인(핏)
바지 길다 > 길이

4.답변은 단답형으로 합니다.
예시
입력: 옷이 이뻐요
출력: 디자인
"""
human_template="다음 문장을 분류해줘, {text}"

create_sentence = Create_Sentence(index_path,"faiss0",template,human_template)
create_sentence.create(text)
```

### 유사도 검색
인접한 k개의 유사한 결과를 추출합니다.
자카드 유사도 임계값은 0.3으로 설정했습니다.
```python
from final_output import Search

search = Search(index_path,"faiss0")
txt = "디자인이 예뻐요"
search.search(10,0.3,txt)
```
