# 104工作內容分析
## 前言

如果有一個學習的目標對於在學的學生無疑是最大的幫助，如果能了解未來工作所需要具備的技能或是能力，想必可以增加學習的動力，而我本身是就讀資料分析相關的領域，因此我對於在104人力銀行上的數據分析職缺進行了分析，希望能透過分析結果了解以下幾點:

- 選擇的工作通常屬於哪種職務類別以及要求的工作技能為何<br>
- 要求的學歷分布<br>
- 要求擅長的工具<br>
- 了解所選工作的工作性質及該具備的能力<br>

## (一)  104人力銀行數據分析職缺內容抓取


```![142761959-5ede1036-ed79-4829-b959-47cce02ec1ed](https://user-images.githubusercontent.com/44692570/142761982-ede34c2b-219f-41b6-8e4e-d8acc22865cd.png)

import job104_spider

# 取得工作內容/job_type=查詢內容/page=查詢頁數
data = job104_spider.get_job_context('數據分析',18)
data.head(5)
```
![image](https://user-images.githubusercontent.com/44692570/142760144-38777eeb-c10a-40d6-a324-c5d42d74a78e.png)

這邊我們抓取了356個工作名稱有提到數據分析的職缺。
## (二) 職務類別&工作技能要求

```
from collections import Counter
import pandas as pd
import numpy as np

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
```
![image](https://user-images.githubusercontent.com/44692570/142760493-85da98e2-b9dd-440f-96a8-aee331219a2a.png)

從資料上來看，可以發現說數據分析的職缺與<b>市場調查與分析</b>這塊息息相關。

## (二) 學歷要求

```
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

import numpy as np
import matplotlib.pyplot as plt
 
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
```
![image](https://user-images.githubusercontent.com/44692570/142760087-ff90a17e-4b4f-4e8e-8a75-e8593ed30477.png)

數據分析這一行對於學歷的要求沒有想像中的高，又或者說會在104上開職缺的工作通常對學歷的要求可能相對會比較沒那麼苛刻。

# (三) 要求擅長的工具的排名

```
import jieba
from collections import Counter

data['擅長工具'] = data.擅長工具.apply(lambda x: x.replace(u'\u200b', '').replace(u'u3000','')) #刪除特殊字符
data['擅長工具2'] = data.擅長工具.apply(lambda x: x.split('、'))  #分詞
tool_text = data.擅長工具2
tool_cnt = Counter([word for sent in tool_text for word in sent])  # 計算次數
tool_fre_most=tool_cnt.most_common(11)[1:] #前10名最常出現的詞

#熱門工具排行長條圖

import numpy as np
import matplotlib.pyplot as plt

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
```
![image](https://user-images.githubusercontent.com/44692570/142760562-3d52f37c-6868-409f-9920-bae306fdc7b0.png)

身為當下熱門程式語言的Python拿下了第一名，超過三分之一的職缺需要會使用Python，而Excel、PowerPoint及Word三樣基本分析工具也在前10名中佔了3個。MS SQL則是打敗其他SQL獨自擠進前10並排序第三。<br>

在前面我們有發現說數據分析的職缺與<b>市場調查與分析</b>這塊有很大的關聯，因此對於製作報表及網路分析的工具也會有不小的需求，而Tableau、Power BI及Google Analytics的上榜也許可以間接證實數據分析與市場調查分析的關聯

# (四) 探討該具備的工具組合

```
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
```
![image](https://user-images.githubusercontent.com/44692570/142761959-5ede1036-ed79-4829-b959-47cce02ec1ed.png)

這裡我使用Apriori演算法去找出數據分析職缺中常出現的擅長工具組合，下面先簡單的解釋下欄位名稱所代表的意思:<br>

- antecedents(前項):類似於前因後果中的原因<br>
- consequents(後項):類似於前因後果中的後果<br>
- support(支持度):以第一筆資料為例，PowerPoint與Excel同時出現在樣本中的機率為0.148876，也就是說356個職缺有53個職缺要求你同時擅長PowerPoint及Excel<br>
- confidence(信賴度):以第一筆資料為例，在要求需要PowerPoint的職位，有100%需要你會Excel<br>
- lift(提升度):計算方式為confidence(A->B)/support(B)，小於1為負相關，等於1為獨立，大於1則是正相關<br>

了解Apriori演算法後來看分析結果，我們可以歸類出下面四種組合，分別為:<br>

### 1. Excel、PowerPoint、Word
### 2. R、Python
### 3. Python、MS SQL
### 4. Tableau、Python 

那會用到第一種組合的工作應該比較偏向文書處理，第二、三種組合對於coding能力的要求相對比較高，第四種組合則偏向注重將數據視覺化的能力。

# (五) 探討工作性質及該具備的能力

```
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
```
![image](https://user-images.githubusercontent.com/44692570/142763302-e91f8212-3122-4112-94c0-3bcd3aab3e4e.png)

這邊我將工作內容的欄位做斷詞，並使用set()函數去除重複的詞，以防一個職缺提到一個詞的頻率太頻繁，而這個詞在其他職缺卻鮮少出現，導致呈現結果不如預期，也就是說若有一個職缺的工作內容提到「大數據的時代，我們需要大數據的人才」，雖然他提到大數據一詞二次，但在我這邊只會記錄一次。<br>

從文字雲圖上可以看到除了<b>報表、資料庫、系統</b>等比較偏硬實力的詞，也可以看到<b>協助、專案、溝通、團隊、合作</b>等與團隊合作相關的詞，因此我們可以得知，與人交際溝通的能力在數據分析相關工作也是非常重要的。另外<b>經驗、熟悉</b>二詞出現的頻率也很高，這蠻好理解，畢竟在我們資訊業經驗的累積與工具的熟悉度是非常重要的，在大學參與產實習或者是產學合作是一個累積經驗與熟悉工具的好方法。

# (六) 若將工作的內容分成3個主題

```
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

lda = LatentDirichletAllocation(n_components=3, max_iter=50,    #n_components=3/分成三個主題
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
```
![image](https://user-images.githubusercontent.com/44692570/142764860-7b5fb994-9f98-4df4-97dc-241692b044a4.png)

這邊使用 scikit-learn的 LatentDirichletAllocation套件將工作內容分成三個主題。主題1像是針對使用者體驗方面，主題2偏向專案的規劃與執行，主題3則是工具的使用方面。

# 總結

除了數據分析職缺以外，我還有另外跑了其他幾個關鍵字，例如資料科學，資料工程等等。經過上面的數據及與其他關鍵字比較後，可以發現數據分析算是資料科學領域中入門門檻比較低的職缺。<br>
就學歷要求而言，資料科學職缺中需要有碩士學歷的比例就遠高於數據分析職缺。資料科學職缺中職務類別的歸類則以演算法開發工程師出現次數最多。再來是工具使用的差異，資料科學職缺熱門使用的工具前10名中已經不見PowerPoint及Word的蹤影，取而代之的則是Java以及Git。而最大的不同則在於工作的內容，數據分析職缺的工作內容著重於資料的分析，資料科學職缺重視模型、演算法的設計，機器學習的能力。若想從事資料科學相關的工作，統計的能力機器學習的功力是不可或缺的<br>

### 那麼以上為個人對於數據分析職缺的一些分析與見解，覺得對你有幫助的麻煩幫忙點個讚👍



