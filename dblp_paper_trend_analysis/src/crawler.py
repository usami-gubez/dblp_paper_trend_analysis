import requests
from bs4 import BeautifulSoup
import time
import os
import pandas as pd
from config import DATA_DIR
from tqdm import tqdm  

CONFERENCES = {
    'cvpr': 'CVPR',
    'icml': 'ICML',
    'iclr': 'ICLR',
    'kdd': 'KDD',
}
YEARS = range(2020, 2025)  
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def get_conference_papers(conf_key, conf_name, year):
    """获取单个会议某年的所有论文"""
    url = f"https://dblp.org/db/conf/{conf_key}/{conf_key}{year}.html"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        print(f"解析网页中({conf_name},{year})......")
        soup = BeautifulSoup(response.text, 'lxml')

        papers = []
        for entry in tqdm(soup.find_all('li', class_=['entry inproceedings', 'entry article']),
                          desc=f"Processing {conf_name} {year}", 
                          unit="paper"):
            paper = {
                'title': entry.find('span', class_='title').text if entry.find('span', class_='title') else None,
                'authors': [a.text for a in entry.find_all('span', itemprop='author')],
                'year': year,
                'conference': conf_name,
                'doi_link': entry.find('a', href=lambda x: x and x.startswith('https://doi.org/'))['href'] 
                           if entry.find('a', href=lambda x: x and x.startswith('https://doi.org/')) else None,
                'dblp_link': entry.find('a', href=lambda x: x and f'rec/conf/{conf_key}' in x)['href'] 
                            if entry.find('a', href=lambda x: x and f'rec/conf/{conf_key}' in x) else None
            }
            papers.append(paper)
        return papers
    except Exception as e:
        print(f"Error fetching {conf_name} {year}: {e}")
        return []

def scrape_all_conferences():
    """爬取所有会议所有年份的数据"""
    all_papers = []
    for conf_key, conf_name in CONFERENCES.items():
        for year in YEARS:
            papers = get_conference_papers(conf_key, conf_name, year)
            all_papers.extend(papers)
            time.sleep(1)  
    return all_papers

print("开始爬取论文数据......")
papers_data = scrape_all_conferences()
print(f"论文数据爬取完成！共爬取到 {len(papers_data)} 篇论文")

df=pd.DataFrame(papers_data)
save_path=os.path.join(DATA_DIR,'raw','dblp_papers_2020-2024.csv')
df.to_csv(save_path,index=False, encoding='utf-8-sig')
print(f"数据已保存至{save_path}")