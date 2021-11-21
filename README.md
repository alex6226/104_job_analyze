# 104工作內容分析
## 前言

有一個學習的方向對於在學的學生無疑是最大的幫助，如果能了解未來工作所需要具備的技能或是能力，想必可以增加學習的動力，而我本身是就讀數據分析相關的領域，因此我對於在104人力銀行上的數據分析職缺進行了分析，希望能透過分析結果了解以下幾點:

- 1.選擇的工作通常屬於哪種職務類別以要求的工作技能為何<br>
- 2.要求的學歷分布<br>
- 3.要求擅長的工具排名<br>
- 4.了解所選工作的工作性質及該具備的能力<br>

## (一)  104人力銀行數據分析職缺內容抓取


```![142761959-5ede1036-ed79-4829-b959-47cce02ec1ed](https://user-images.githubusercontent.com/44692570/142761982-ede34c2b-219f-41b6-8e4e-d8acc22865cd.png)

import job104_spider

# 取得工作內容/job_type=查詢內容/page=查詢頁數
data = job104_spider.get_job_context('數據分析',18)
data.head(5)
```
![image](https://user-images.githubusercontent.com/44692570/142760144-38777eeb-c10a-40d6-a324-c5d42d74a78e.png)

這邊我們抓取了356個工作名稱有提到數據分析的職缺
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

從資料上來看，可以發現說數據分析的職缺與<b>市場調查與分析</b>這塊息息相關

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

數據分析這一行對於學歷的要求沒有想像中的高，又或者說會在104上開職缺的工作通常對學歷的要求可能相對會比較沒那麼苛刻

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

了解Apriori演算法後我們來看分析結果，我們可以歸類出四種組合，分別為:<br>

### 1.Excel、PowerPoint、Word
### 2.R、Python
### 3.Python、MS SQL
### 4.Tableau、Python 








