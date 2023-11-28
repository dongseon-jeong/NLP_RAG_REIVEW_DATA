# RAG

## langchain
파이프라인  
chat  
agent  
memory  
생성 테스트  

## 임베딩 모델 선택
허깅페이스모델  
klue/roberta-small  
kt믿음  

오픈ai 모델  
text-embedding-ada-002  

## 리트리버
문서 추출  
텍스트 스플리터    

## 벡터 DB
파인콘  
faiss  
elestic search  

## 인덱스 생성
저장  

## RAG
활용  
main.py 코드입니다.  
langchain과 openai 임베딩을 사용했습니다.


전처리 테이블 생성
```
from preprocessing import review_tag, re90

review_tag(in_path,out_path,date)
re90(in_path,out_path)
```

임베딩 생성
```
from review_embedding import Review_Embedding

review_embedding = Review_Embedding(out_path)
review_embedding.create_dataset(out_path)
review_embedding.create_index(index_path)
```

문장 생성
```
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

키워드 분류
```
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

유사도 검색

```
from final_output import Search

search = Search(index_path,"faiss0")
txt = "디자인이 예뻐요"
search.search(10,0.3,txt)
```
