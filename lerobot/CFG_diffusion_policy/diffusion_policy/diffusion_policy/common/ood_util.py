import numpy as np

def find_threshold_by_percentile(data, percentile=99):
    """
    基于经验CDF选择阈值
    percentile: 百分位数 (0-100)
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    print(sorted_data[1500])
    # 计算指定百分位数的索引
    index = int(np.ceil(percentile / 100 * n)) - 1
    print(index)
    index = max(0, min(index, n - 1))
    
    threshold = sorted_data[index]
    return threshold