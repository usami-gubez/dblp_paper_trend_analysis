import os
from config import DATA_DIR
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def get_paper_count():
    """获得每个会议每年论文的数量"""
    data_df=pd.read_csv(os.path.join(DATA_DIR,'processed','dblp_papers_cleaned.csv'))
    paper_count_df = data_df.groupby(['conference', 'year']).size().reset_index(name='count')
    return paper_count_df

def predict_for_conference(conference,df):
    conf_data = df[df['conference']==conference].sort_values('year')
    years = conf_data['year'].values.reshape(-1,1)
    counts = conf_data['count'].values
    
    if conference in ['CVPR', 'ICML']:
        # 二次回归
        X = np.column_stack([years, years**2])
        model = LinearRegression().fit(X, counts)
        pred = int(model.predict([[2025, 2025**2]])[0])
        
    elif conference == 'ICLR':
        # Holt's指数平滑
        model = ExponentialSmoothing(counts, trend='add', damped_trend=True).fit()
        pred = int(model.forecast(1)[0])
        
    else:  # KDD
        # 线性增量
        diffs = np.diff(counts)
        pred = counts[-1] + int(np.mean(diffs))
    
    return pred

def predict_next_year(df):
    conferences = df['conference'].unique()
    predictions = []
    for conf in conferences:
        pred_count = predict_for_conference(conf,df)
        predictions.append({'conference': conf, 'year': 2025, 'count': pred_count})
    result_df = pd.concat([df, pd.DataFrame(predictions)], ignore_index=True)
    return result_df

print("正在预测论文数量趋势......")
df=get_paper_count()
PREDIECTED_PAPER_COUNT_DF=predict_next_year(df)
print("论文数量趋势预测完成！")