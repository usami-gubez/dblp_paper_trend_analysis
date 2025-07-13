import pandas as pd
import re
import spacy
from tqdm import tqdm
import os
from config import DATA_DIR

def extract_meaningful_phrases(doc):
    phrases = []
    # 规则1：提取名词短语（含形容词+名词组合）
    for chunk in doc.noun_chunks:
        phrases.append(chunk.text.lower())
    
    # 规则2：提取动名词结构
    for token in doc:
        if token.tag_ in ["VBG", "NN"] and token.dep_ in ["compound", "amod"]:
            phrase = " ".join([t.text for t in token.subtree]).lower()
            phrases.append(phrase)
    
    # 规则3：提取特定词性组合
    for i in range(len(doc)-1):
        if doc[i].pos_ in ["ADJ", "NOUN"] and doc[i+1].pos_ == "NOUN":
            phrases.append(f"{doc[i].text} {doc[i+1].text}".lower())
    
    for phrase in list(set(phrases)):  
        words = phrase.split()
        if len(words)>1:
            for i in range(len(words)-1):         
                for j in range(i + 2, len(words) + 1):  
                    phrase = " ".join(words[i:j])
                    phrases.append(phrase)

    return list(set(phrases))  

def batch_process(texts,nlp):
    filtered_phrases_list = []
    for doc in tqdm(nlp.pipe(texts, batch_size=500), total=len(texts)):
        filtered_phrases_list.append(extract_meaningful_phrases(doc)) # 调入处理好的doc对象
    return filtered_phrases_list

def custom_tokenizer(nlp):
    tokenizer = nlp.tokenizer
    tokenizer.token_match = re.compile(r'^[-\w]+$').match
    return tokenizer

conference_blacklist = {
    'CVPR': {
        'a single',
        'a large-scale',
        'a unified',
        'single image',
    },
    'ICLR': {
        'a simple',
        'a general',
        'the role',
        'and efficient',
        'an efficient',
    },
    'ICML': {
        'a simple',
        'a single',
        'a unified',
        'differentially private'
    },
    'KDD': {
        'a fast',
        'a novel',
        'an efficient',
        'fast and',
        'and accurate',
        'fast and accurate',
        'a survey',
        'efficient and',
        'large scale',
        'case study',
        'and efficient',
        'and interpretable',
        'learning framework',
        'novel approach'
    }
}

base_blacklist={
    'language model',
    'large language',
    'time series',
    'diffusion model',
    'point cloud',
    'radiance field',
    'neural network',
    'learning representation',
    'foundation model',
    'offline reinforcement',
    'graph neural',
    'generative model',
    'deep reinforcement',
    'gradient descent',
    'gradient policy',
    'vision language',
    'anomaly detection',
}

def filter_meaningless_phrases(df, conference):
    conference_specific = conference_blacklist.get(conference, set())
    combined_blacklist = conference_specific | base_blacklist 
    
    filtered_df = df[
        (~df['filtered_phrases'].isin(combined_blacklist)) & 
        (df['filtered_phrases'].str.split().str.len() > 1)
    ]
    return filtered_df

def merge_similar_phrases_df(df):
    phrase_mapping = {}
    all_phrases = set(df['filtered_phrases'].unique())
    for phrase in all_phrases:
        base_phrase = phrase.lower().strip()
        
        # 规则1: 处理单复数
        if base_phrase.endswith('s'):
            singular = base_phrase[:-1]
            if singular in all_phrases:
                base_phrase = singular
        elif (base_phrase + 's') in all_phrases:
            base_phrase = base_phrase  
        
        # 规则2: 处理连字符/空格变体
        if '-' in base_phrase:
            spaced = base_phrase.replace('-', ' ')
            if spaced in all_phrases:
                base_phrase = spaced
        elif ' ' in base_phrase:
            hyphenated = base_phrase.replace(' ', '-')
            if hyphenated in all_phrases:
                base_phrase = hyphenated
        
        # 规则3: 处理冠词 (a/an/the)
        base_phrase = re.sub(r'^(a|an|the)\s+', '', base_phrase)
        
        # 规则4: 处理不同词序 (如 "object 3d" 和 "3d object")
        words = base_phrase.split()
        if len(words) == 2:
            reversed_phrase = f"{words[1]} {words[0]}"
            if reversed_phrase in all_phrases:
                # 选择字母顺序靠前的作为基础形式
                base_phrase = min(base_phrase, reversed_phrase)
        
        phrase_mapping[phrase] = base_phrase
    df['base_phrase'] = df['filtered_phrases'].map(phrase_mapping)
    merged_df = df.groupby(['conference', 'year', 'base_phrase'])['count'].sum().reset_index()
    merged_df = merged_df.rename(columns={'base_phrase': 'filtered_phrases'})
    merged_df = merged_df.sort_values(['conference', 'year', 'count'], ascending=[True, True, False])
    return merged_df.reset_index(drop=True)

def main():
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv(os.path.join(DATA_DIR,'processed','dblp_papers_cleaned.csv'))
    nlp.tokenizer = custom_tokenizer(nlp)
    print('正在分析论文标题关键词频率......')
    # 提取所有短语（不进行过滤）
    df['all_phrases'] = batch_process(df['title'].tolist(), nlp)
    # 展开并统计初步频率
    grouped = df.explode('all_phrases').groupby(
        ['conference', 'year', 'all_phrases']).size().reset_index(name='count')
    # 合并相似短语
    merged_df = merge_similar_phrases_df(grouped.rename(columns={'all_phrases': 'filtered_phrases'}))
    # 按会议过滤无意义短语
    final_dfs = []
    for conf in merged_df['conference'].unique():
        conf_df = merged_df[merged_df['conference'] == conf]
        filtered_conf_df = filter_meaningless_phrases(conf_df, conf)
        final_dfs.append(filtered_conf_df)
    result_df = pd.concat(final_dfs).sort_values(['conference', 'year', 'count'], ascending=[True, True, False])
    print('论文标题关键词频率分析完成！')
    return result_df


PAPER_KEYWORD_COUNT_DF=main()
save_path=os.path.join(DATA_DIR,'processed','original_keyword_count.csv')
PAPER_KEYWORD_COUNT_DF.to_csv(save_path)
print(f'数据已经保存到{save_path}')