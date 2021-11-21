# 104工作內容分析
## 前言

有一個學習的方向對於在學的學生無疑是最大的幫助，如果能了解未來工作所需要具備的技能或是能力，想必可以增加學習的動力，而我本身是就讀數據分析相關的領域，因此我對於在104人力銀行上的數據分析職缺進行了分析，希望能透過分析結果了解以下幾點:

- 1.選擇的工作通常屬於哪種職務類別以要求的工作技能為何<br>
- 2.要求的學歷分布<br>
- 3.要求擅長的工具排名<br>
- 4.了解所選工作的工作性質及該具備的能力<br>

## (一)  104人力銀行職缺內容抓取


```
import job104_spider

# 取得工作內容/job_type=查詢內容/page=查詢頁數
data = job104_spider.get_job_context('數據分析',18)
data.head(5)
```
![image](https://user-images.githubusercontent.com/44692570/142759115-8f22eeb7-6d20-4dfe-9e39-deae7c6980f4.png)

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


