import pandas as pd

## 리뷰 파일
def review_tag(in_path,out_path,date):
    review = pd.read_csv(in_path+'reviews.csv')
    review = review[["review_id","brand_code","review_created_at"]]
    review_tags = pd.read_csv(in_path+'review_tags.csv')
    review_tags = review_tags.merge(review,on = ['review_id','brand_code'])
    review_tags = review_tags[review_tags['review_created_at']>= date]
    review_tags = review_tags.drop(labels='review_created_at', axis= 1)
    review_tags = review_tags.reset_index(drop=True)
    review_tags.to_csv(out_path+"review_tags.csv")


## 명사추출하기
def re90(in_path,out_path):
    df = pd.read_csv(in_path+'tokens.csv')
    search_string = "^NN"
    result = df[df["pos"].str.contains(search_string)]
    re = result.groupby('token')['token'].count().reset_index(name='counts')
    re.sort_values(by=['counts'], ascending=False, inplace=True)
    re = re.reset_index(drop=True)
    print(f"총 명사 갯수: {len(re)}")
    print(f" 1개 명사 갯수: {re[re['counts']==1]['token'].count()} 비율: {re[re['counts']==1]['token'].count()/len(re)*100:.1f}%")
    # 명사별 언급비중 계산
    re['perc'] = (re['counts']/sum(re['counts'])*100)
    re['running_total'] = pd.Series(re['perc']).cumsum(skipna=False)
    re90 = re[re['running_total']<=90]['token']
    re90.to_csv(out_path+"re90.csv",index=False)




