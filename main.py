import sys
import pandas as pd
import os
sys.path.append('C:\\Users\\jeong\\langchain\\script')
from preprocessing import review_tag, re90
from review_embedding import Review_Embedding
from sentence_create import Create_Sentence
from final_output import Search


## 변수 설정
in_path = "\\output\\"
out_path = "\\test\\"
index_path = "\\test2\\"
date = "2023-09-25"
# model = ""

## api 키 설정
OPENAI_API_KEY = "sk-" #키입력
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


## 1 데이터 준비
# 리뷰텍 기간 한정 추출 후 저장
review_tag(in_path,out_path,date)
# 명사 언급수 90% 추출
re90(in_path,out_path)


## 2 리뷰 임베딩(허깅페이스 모델 코드 Review_Embedding_HU)
# 코드 불러오기
review_embedding = Review_Embedding(out_path)
# print(f"10만단위 개수: {review_embedding.len()}")
# print(f"10만단위 리스트: {review_embedding.len_list()}")
# 10만개 이상의 경우 api 호출 시 램용량 초과 현상이 있어 10만개씩 분해하는 과정
review_embedding.create_dataset(out_path)
# 각 데이터셋에 faiss 각 인덱스 생성 후 저장(벡터로 변경, 10만개 이상일 경우 인덱스 merge)
review_embedding.create_index(index_path)


## 3 생성
nn = pd.read_csv(out_path+"re90.csv")
text = nn['token'][0]
template="""
1.당신은 명사 다음에 올 동사나 형용사를 구하는 ai입니다.

2.최종 대표적인 동사, 형용사 최대 4개를 추출합니다.

3.형용사나 동사는 '다'로 끝나는 형태로 출력합니다.
예시)
짧은, 짧아서 > 짧다
길어서, 긴 > 길다

4.답변은 단답형으로 합니다.
예시)
입력: 길이
출력: 짧다,길다
"""
human_template="다음 명사 다음 단어를 구해줘, {text}"

create_sentence = Create_Sentence(index_path,"faiss0",template,human_template)
create_sentence.create_rag(text)


## 4 키워드 분류
text = "예뻐요"
template="""
1.당신은 키워드를 분류하는 ai입니다.

2.키워드 14개이고, 다음과 같습니다.
'가격','기능성','길이','디자인','라인(핏)','마감처리','배송','사이즈','색상','소재','스타일','신축성','착용감','품질'

3.가장 유사한 의미를 나타내는 키워드로 분류합니다.
예시)
몸에 맞다 > 라인(핏)
바지 길다 > 길이

4.답변은 단답형으로 합니다.
예시)
입력: 옷이 이뻐요
출력: 디자인
"""
human_template="다음 문장을 분류해줘, {text}"

create_sentence = Create_Sentence(index_path,"faiss0",template,human_template)
create_sentence.create(text)


## 5 유사 문장 찾기
search = Search(index_path,"faiss0")
txt = "디자인이 예뻐요"
search.search(10,0.3,txt)