# dblp_paper_trend_analysis

## 项目介绍

该项目实现了对CVPR、ICML、ICLR、KDD四个会议研究趋势以及未来刊登论文数量趋势的分析与可视化

项目首先从DBLP上爬取了四个会议近五年的所有论文的标题，接着对这些论文的数量以及标题的关键词进行了分析研究，最后对结果进行了可视化

## 环境依赖

python 3.10

包：requests、bs4、pandas、 tqdm、re、spacy、numpy、sklearn、statsmodels、wordcloud、collections、matplotlib

## 项目目录

```bash
│   READEME.md
│
├───data
│   ├───processed  # 存放处理过的数据
│   └───raw        # 存放初始爬取的数据
├───result
│   ├───keyword_analysis    # 存放关键词分析的可视化结果
│   │   ├───frequency       # 频率分析
│   │   └───trend           # 趋势分析
│   └───paper_count_trend   # 存放论文数量趋势分析的可视化结果
└───src
    │   config.py   # 项目路径信息  
    │   crawler.py  # 爬虫
    │   data_processing_base.py         # 基础数据清洗
    │   keyword_frequency_analysis.py   # 关键词频率分析
    │   keyword_trend_analysis.py       # 研究趋势分析
    │   paper_count_trend_analysis.py   # 论文数量趋势分析
    └── visualization.py                # 可视化
```

## 使用说明

1. 运行crawler.py爬取数据
2. 运行data_processing_base.py清洗数据
3. 运行keyword_frequency_analysis.py获得关键词频率数据
4. 运行visualization.py（其会调用另外两个分析代码）
