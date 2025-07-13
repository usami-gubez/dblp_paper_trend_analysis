import os
import pandas as pd
import numpy as np
from config import DATA_DIR
from tqdm import tqdm

def calculate_hot_index(alpha):
    df = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'original_keyword_count.csv'))
    
    # 为每个会议-关键词组合计算热门指数
    def compute_conference_keyword_hot_index(group):
        # 只关注2024年的关键词
        keywords_2024 = group[group['year'] == 2024]['filtered_phrases'].unique()
        results = []
        for keyword in tqdm(keywords_2024):
            keyword_data = group[group['filtered_phrases'] == keyword]
            years = range(2020, 2025)
            counts = []
            for y in years:
                count = keyword_data[keyword_data['year'] == y]['count'].sum()
                counts.append(count if not np.isnan(count) else 0)
            
            # 计算上升指数 (使用线性回归的斜率)
            x = np.array(years)
            y = np.array(counts)
            if len(y) > 1: 
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0
            
            # 计算热门指数 (上升指数 * 2024年出现次数)
            hot_index = (1 + alpha*slope/np.abs(slope).max()) * counts[-1]
            results.append({
                'conference': group['conference'].iloc[0],
                'keyword': keyword,
                'hot_index': hot_index
            })
        
        return pd.DataFrame(results)
    
    # 对每个会议应用计算
    result_df = df.groupby('conference').apply(compute_conference_keyword_hot_index).reset_index(drop=True)
    
    final_result = result_df.sort_values(['conference', 'hot_index'], ascending=[True, False]) \
                          .groupby('conference') \
                          .head(200) \
                          .reset_index(drop=True)
    
    return final_result

hot_keywords_df = calculate_hot_index(1.7)
save_path=os.path.join(DATA_DIR,'processed','hot_keyword.csv')
hot_keywords_df.to_csv(save_path)
print(f"数据已经保存到{save_path}")
