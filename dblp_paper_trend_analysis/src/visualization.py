import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import RESULT_DIR,DATA_DIR
from paper_count_trend_analysis import PREDIECTED_PAPER_COUNT_DF
import pandas as pd
from wordcloud import WordCloud
from collections import defaultdict
from tqdm.auto import tqdm
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

def plot_paper_count():
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=PREDIECTED_PAPER_COUNT_DF,
        x='year',
        y='count',
        hue='conference',
        marker='o',  
        linewidth=2.5,
        markersize=8,
    )
    plt.title('各个会议在2020年至2024年刊登的论文数量（2025年为预测值）', fontsize=14)
    plt.xlabel('年份', fontsize=12)
    plt.ylabel('论文数量', fontsize=12)
    plt.legend(title='会议')
    plt.grid(True, linestyle='--', alpha=0.6)
    save_path=os.path.join(RESULT_DIR,'paper_count_trend','会议-年份-数量.png')
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

def process_filtered_phrases(df):
    grouped = df.groupby(['year', 'conference'])
    def get_top_records(group):
        sorted_group = group.sort_values('count', ascending=False)
        top_rows = max(1, int(len(sorted_group) * 0.01))
        top_rows = min(top_rows, 50)
        return sorted_group.head(top_rows)
    result = grouped.apply(get_top_records).reset_index(drop=True)
    return result

def plot_keyword_frequency():
    df=pd.read_csv(os.path.join(DATA_DIR,'processed','original_keyword_count.csv'))
    filter_df=process_filtered_phrases(df)
    grouped = filter_df.groupby(['year', 'conference'])
    word_freq_dicts = defaultdict(dict)
    
    for (year, conf), group in grouped:
        for _, row in group.iterrows():
            phrase = row['filtered_phrases']
            count = row['count']
            word_freq_dicts[(year, conf)][phrase] = count
    
    for (year, conf), freq_dict in tqdm(word_freq_dicts.items()):
        wc = WordCloud(
            width=1200,
            height=600,
            background_color="#678a93",
            colormap='plasma',
            max_words=200,
            prefer_horizontal=0.9,  # 增加横排概率（0-1）
            collocations=False,      # 禁用词组关联
            scale=2,                 # 提高计算精度
            relative_scaling=0.5,    # 平衡大小差异（0-1）
            min_font_size=4,         # 避免过小字体
            margin=2,               # 减小边距
            random_state=42,          # 固定随机种子保证可重复
            font_path='c:\WINDOWS\Fonts\BAHNSCHRIFT.TTF'
        ).generate_from_frequencies(freq_dict)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(f"会议: {conf} ({year})\nTop关键词", fontsize=14)
        plt.axis("off")
        
        filename = f"词云图_{conf}_{year}.png"
        save_path=os.path.join(RESULT_DIR,'keyword_analysis','frequency',filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_conference_keywords(colormap):
    df=pd.read_csv(os.path.join(DATA_DIR,'processed','hot_keyword.csv'))
    conferences = df['conference'].unique()
    
    for conf in conferences:
        conf_data = df[df['conference'] == conf]
        sorted_data = conf_data.sort_values("hot_index", ascending=False).head(20)
        
        if len(sorted_data) == 0:
            continue  
            
        plt.figure(figsize=(10, 7))
        gradient = np.linspace(0, 1, len(sorted_data))
        bar_colors = plt.get_cmap(colormap)(gradient)
        
        bars = plt.barh(
            sorted_data['keyword'],
            sorted_data['hot_index'],
            color=bar_colors
        )
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width,
                    bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}',  # 保留1位小数
                    va='center', ha='left',
                    fontsize=10)
        
        # 图表装饰
        plt.title(f"{conf} - 热门关键词TOP20", fontsize=16, pad=20)
        plt.xlabel("热度指数 (hot_index)", fontsize=12)
        plt.ylabel("关键词", fontsize=12)
        plt.gca().invert_yaxis() 
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        safe_conf_name = "".join(c if c.isalnum() else "_" for c in conf)
        filename = f"{safe_conf_name}_关键词热度.png"
        
        save_path=os.path.join(RESULT_DIR,'keyword_analysis','trend')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
        plt.close()  


print('正在生成原始频率词云图......')
plot_keyword_frequency()
print('词云图生成完成！')
plot_paper_count()
colors = [(0.3, 0.1, 0.5), (0.8, 0.6, 1.0)]  # 深紫→浅紫
cmap = LinearSegmentedColormap.from_list("edu", colors)
plot_conference_keywords(cmap)