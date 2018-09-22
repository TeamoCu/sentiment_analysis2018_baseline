#!/user/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import jieba
import config
import io

jieba.load_userdict(config.dict_path)
# 加载数据
def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df


# 分词
def seg_words(contents):
    contents_segs = list()
    for content in contents:
        segs = jieba.lcut(content)
        contents_segs.append(" ".join(segs))

    return contents_segs


if __name__ == '__main__':
    datadf = load_data_from_csv(config.train_data_path)
    traf_data = datadf.iloc[0:config.train_data_size, [1, 2]]
    traf_data = traf_data.loc[
        (traf_data["location_traffic_convenience"] == 1) | (traf_data["location_traffic_convenience"] == -1)]
    # # traf_data.loc[
    # traf_data["location_traffic_convenience"] == 1 or traf_data["location_traffic_convenience"] == -1, ["content",
    #                                                                                                      "location_traffic_convenience"]]
    print traf_data
