import job104_spider    
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba
 
data = job104_spider.get_job_context('數據分析', 18)  # 獲取數據

'''--------------------------------------職務類別與工作技能排名--------------------------------------'''

#職務類別排名

data['職務類別2'] = data.職務類別.apply(lambda x: x.split('、')) #分詞
category_test = data.職務類別2
category_cnt = Counter([word for sent in category_test for word in sent])  # 計算次數
category_fre_most=category_cnt.most_common(10) #顯示前10名最常出現的詞
Job_category_df=pd.DataFrame({'職務類別': [key for key, values in category_fre_most],
              '出現次數': [values for key, values in category_fre_most]},index=np.arange(1,11))

#工作技能排名

data['工作技能2'] = data.工作技能.apply(lambda x: x.split('、'))
Job_ability_text = data.工作技能2
Job_ability_text_cnt = Counter([word for sent in Job_ability_text for word in sent])  # 計算次數
Job_ability_fre_most=Job_ability_text_cnt.most_common(11)[1:]    #第一個為''
Job_ability_df=pd.DataFrame({'工作技能': [key for key, values in Job_ability_fre_most],
              '出現次數': [values for key, values in Job_ability_fre_most]},index=np.arange(1,11))

df_concat=pd.concat([Job_category_df,Job_ability_df],axis=1)
df_concat

'''----------------------------------------最低學歷要求分布----------------------------------------'''


#最低學歷要求分布

def Education_require(sentence):
    
    if '不拘' in sentence:
        return '不拘'

    elif '高中' in sentence:
        return '高中'

    elif '專科' in sentence or '大學' in sentence:    
        return '專科、大學'
    
    elif '碩士' in sentence:
        return '碩士'
    
    elif '博士' in sentence:
        return '博士'

data['最低學歷要求']=data.學歷要求.apply(lambda x :Education_require(x))


#最低學歷要求分布長條圖
 
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta'] 
height = data.最低學歷要求.value_counts().values
bars = data.最低學歷要求.value_counts().index
x_pos = np.arange(len(bars))

plt.title('最低學歷要求')
plt.bar(x_pos, height, color='lightblue',  edgecolor='blue')
plt.xticks(x_pos, bars)
plt.ylabel('出現次數')

for x, y in enumerate(height):
    plt.text(x, y, '%s' % y, ha='center',va='bottom')

plt.show()

'''-----------------------------------------熱門工具排行-----------------------------------------'''

data['擅長工具'] = data.擅長工具.apply(lambda x: x.replace(u'\u200b', '').replace(u'u3000','')) #刪除特殊字符
data['擅長工具2'] = data.擅長工具.apply(lambda x: x.split('、'))  #分詞
tool_text = data.擅長工具2
tool_cnt = Counter([word for sent in tool_text for word in sent])  # 計算次數
tool_fre_most=tool_cnt.most_common(11)[1:] #前10名最常出現的詞

#熱門工具排行長條圖

plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
height = [values for key, values in tool_fre_most]
bars = [key for key, values in tool_fre_most]
x_pos = np.arange(len(bars))
plt.figure(figsize=(10, 4))
plt.title('熱門工具排名')
plt.bar(x_pos, height, color='#9999FF')
plt.xticks(x_pos, bars, rotation=45)
plt.ylabel('出現次數')

for x, y in enumerate(height):
    plt.text(x, y, '%s' % y, ha='center',va='bottom')

plt.show()

'''-----------------------------使用關聯規則找出容易同時需要具備的工具-----------------------------'''

#使用關聯規則找出容易同時需要具備的工具

from mlxtend.preprocessing import TransactionEncoder    #編碼器
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

te = TransactionEncoder()
te_ary = te.fit(data['擅長工具2']).transform(data['擅長工具2'])
tool_df = pd.DataFrame(te_ary, columns=te.columns_)
tool_df.head(5)


frequent_items = apriori(tool_df, min_support=0.1, use_colnames=True)   #min_support為0.1
association_df = association_rules(frequent_items,metric='confidence', min_threshold=0.3).sort_values(by='confidence', ascending=False)[
    ['antecedents', 'consequents', 'support', 'confidence', 'lift']].reset_index(drop=True) #min_confidence為0.3

association_df

'''------------------------------------利用文字雲找出工作內容注重的能力------------------------------------'''
#文字雲

import jieba
import re

stopwords = [k.strip() for k in open('停用詞.txt', encoding='utf-8') if k.strip() != '']    #停用詞

jieba.set_dictionary('dict.txt.big')  # 繁體中文檔

#將工具的專有名詞加入自定義詞
for keys in tool_cnt.keys():
    jieba.add_word(keys)

#刪除標標點符號
def clean_Punctuation(text):
    text=re.sub(r'[^\w\s]','',text) #刪除標標點符號
    text=text.replace('\n','').replace('\r','').replace('\t','').replace(' ','').replace('[','').replace(']','')
    return text
#斷詞/去除標點符號/去除停用字
def text_cut(sentence):
    sentence_cut=[word for word in jieba.lcut(clean_Punctuation(sentence)) if word not in stopwords]

    return sentence_cut

job_content= data.工作內容 
job_content_cut=[word for sent in job_content for word in set(text_cut(sent)) if word not in '數據分析']  #斷詞
job_content_cut_cnt=Counter([word for word in job_content_cut])   #計算次數

#文字雲作圖

import wordcloud # 詞雲展示庫
from PIL import Image # 影像處理庫
import matplotlib.pyplot as plt # 影像展示庫


#根據單詞及其頻率生成詞雲
wc = wordcloud.WordCloud(
    font_path='NotoSerifCJKtc-Medium.otf', # 設定字型格式
    max_words=40, # 最多顯示詞數
    max_font_size=180, # 字型最大值
    background_color='white',
    width=800, height=600,
)

wc.generate_from_frequencies(job_content_cut_cnt) # 從字典生成詞雲
plt.imshow(wc) # 顯示詞雲
plt.axis('off') # 關閉座標軸
plt.show() # 顯示影像

'''對工作內容進行主題分析'''
#主題分析

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 將字串切割後依照逗點分割
data['工作內容2'] = data.工作內容.apply(lambda x: str(text_cut(x)).replace(
    '[', '').replace(']"', '').replace(',', '').replace('\'', ''))

     
data['其他條件2'] = data.其他條件.apply(lambda x: str(text_cut(x)).replace(
    '[', '').replace(']"', '').replace(',', '').replace('\'', ''))

#模型訓練
tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                max_df=0.5,
                                min_df=10)  
tf = tf_vectorizer.fit_transform(data['工作內容2'])

lda = LatentDirichletAllocation(n_components=3, max_iter=50,    #n_components=5/分成五個主題
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

lda.fit(tf)

def print_topic(feature_names, n_top_words):
    for topic_idx, topic in enumerate(lda.components_):

        print("主題 %d:" % (topic_idx+1))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()
    
tf_feature_names = tf_vectorizer.get_feature_names()
print_topic(tf_feature_names, 20)