from config import DATA_DIR
import os
import pandas as pd
import re

raw_data_df=pd.read_csv(os.path.join(DATA_DIR,'raw','dblp_papers_2020-2024.csv'))

def clean_text(text):
    if not isinstance(text,str):
        return ""
    text = re.sub(r'[^\w\s-]', '', text)
    # 合并连续空格
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

raw_data_df['title']=raw_data_df['title'].apply(clean_text)
pro_data_df=raw_data_df[['title','year','conference']]

pro_data_df.to_csv(os.path.join(DATA_DIR,'processed','dblp_papers_cleaned.csv'))
