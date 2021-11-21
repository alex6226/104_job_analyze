from bs4 import BeautifulSoup
import requests
import pandas as pd

 

# 取得工作連結
def get_href(job_type, page):

    href_list = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) "
               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36"}

    for i in range(1, page+1):

        url = f'https://www.104.com.tw/jobs/search/?ro=0&kwop=1&keyword={job_type}&expansionType=job&order=14&asc=0&page={i}&mode=s&langFlag=0' #kwop=1/只抓包含關鍵字相同的工作
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, "lxml")
        a_list = soup.find_all('a', 'js-job-link')
        for j in a_list:
            href = 'https:'+j['href']
            if 'relevance' in href:
                href_list.append(href)

    return href_list

 

# 取得工作內容/job_type=查詢內容/page=查詢頁數
def get_job_context(job_type, page):
    href_list = get_href(job_type, page)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.92 Safari/537.36',
        'Referer': 'https://www.104.com.tw/job/'  # 必須提供
    }

    temp = []  # 存放df
    for href in href_list:

        replace_word = href[-11:-10]  # 替換字元
        replace__text = 'jobsource=jolist_'+replace_word+'_relevance'
        id = href.replace('https://www.104.com.tw/job/',
                          '').replace(replace__text, '')
        href = f'https://www.104.com.tw/job/ajax/content/{id}'  # 替換成ajax格式

        r = requests.get(href, headers=headers)
        soup = BeautifulSoup(r.text, "lxml")
        json_data = r.json()

        category = ''
        for i in json_data['data']['jobDetail']['jobCategory']:  # 職務類別
            category += ('、'+i['description'])

        skill = ''
        for i in json_data['data']['condition']['skill']:  # 工作技能
            skill += ('、'+i['description'])

        tool = ''
        for i in json_data['data']['condition']['specialty']:  # 擅長工具
            tool += ('、'+i['description'])

        df = pd.DataFrame({'職務名稱': json_data['data']['header']['jobName'],
                           '工作內容': json_data['data']['jobDetail']['jobDescription'],
                           '職務類別': category[1:],
                           '學歷要求': json_data['data']['condition']['edu'],
                           '工作經歷': json_data['data']['condition']['workExp'],
                           '工作技能': skill[1:],
                           '擅長工具': tool[1:],
                           '其他條件': json_data['data']['condition']['other']
                           }, index=[0])
        temp.append(df)

    all_df = pd.concat(temp).reset_index(drop=True)

    return all_df
