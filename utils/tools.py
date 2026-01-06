import logging
import numpy as np
from dateutil.relativedelta import relativedelta
import datetime
import pandas as pd


def setup_logging(log_file):
    """
    配置日志记录器，将日志输出到文件和标准输出。

    Args:
        log_file (str): 日志文件的路径。

    Returns:
        logging.Logger: 配置好的日志记录器对象。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger()


def generate_date_list(start_time, end_time):
    """
    生成从起始日期到结束日期的日期列表。

    Args:
        start_time (int): 起始日期，格式为YYYYMMDD。
        end_time (int): 结束日期，格式为YYYYMMDD。

    Returns:
        list[int]: 日期列表，格式为[YYYYMMDD]。
    """
    start = datetime.datetime.strptime(str(start_time), "%Y%m%d")
    end = datetime.datetime.strptime(str(end_time), "%Y%m%d")
    return [
        int(dt.strftime("%Y%m%d"))
        for dt in [start + relativedelta(days=i) for i in range((end - start).days + 1)]
    ]


def time_features(dates):
    """
    从日期列表中提取归一化的时间特征（月、日），归一化至[-0.5, 0.5]。

    Args:
        dates (list[int]): 日期列表，格式为YYYYMMDD。

    Returns:
        np.ndarray: 提取的时间特征数组，归一化。
    """
    dates = pd.to_datetime(dates, format="%Y%m%d")

    # 计算 month 和 day 特征
    month_feature = (dates.month - 1) / 11.0 - 0.5
    day_feature = (dates.day - 1) / 30.0 - 0.5

    # 按列堆叠特征
    return np.column_stack([month_feature, day_feature])


def prepare_input_target_indices(
    time_length,
    input_gap,
    input_length,
    prediction_shift,
    prediction_gap,
    prediction_length,
    sample_gap,
):
    """
    准备输入和目标样本的索引。

    Args:
        time_length (int): 时间序列的总长度。
        input_gap (int): 两个连续输入帧之间的时间间隔。
        input_length (int): 输入帧的数量。
        prediction_shift (int): 最后一个目标预测的前导时间。
        prediction_gap (int): 两个连续输出帧之间的时间间隔。
        prediction_length (int): 输出帧的数量。
        sample_gap (int): 两个检索样本的起始时间之间的间隔。

    Returns:
        tuple[np.ndarray, np.ndarray]: 输入样本索引和目标样本索引。
    """
    assert prediction_shift >= prediction_length
    input_span = input_gap * (input_length - 1) + 1
    input_index = np.arange(0, input_span, input_gap)
    target_index = (
        np.arange(0, prediction_shift, prediction_gap) + input_span + prediction_gap - 1
    )
    indices = np.concatenate([input_index, target_index]).reshape(
        1, input_length + prediction_length
    )
    max_sample_count = time_length - (input_span + prediction_shift - 1)
    indices = indices + np.arange(max_sample_count)[:, np.newaxis] @ np.ones(
        (1, input_length + prediction_length), dtype=int
    )
    input_indices = indices[::sample_gap, :input_length]
    target_indices = indices[::sample_gap, input_length:]
    assert len(input_indices) == len(target_indices)
    return input_indices, target_indices
