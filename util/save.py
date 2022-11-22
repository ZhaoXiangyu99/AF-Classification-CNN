from typing import List, Tuple
import csv
import scipy.io as sio
import numpy as np
import os

def save_predictions(predictions: List[Tuple[str, str, any]], folder: str = None) -> None:
    """
    将给定的预测保存到名为 PREDICTIONS.csv 的 CSV 文件中
    Parameters
    ----------
    predictions : List[Tuple[str, str,float]]
        List and Tuple，其中每个元组包含文件名和预测标签('N','A','O','~'），以及不确定性
        比如 [('train_ecg_03183.mat', 'N'), ('train_ecg_03184.mat', "~"), ('train_ecg_03185.mat', 'A'),
                  ('train_ecg_03186.mat', 'N'), ('train_ecg_03187.mat', 'O')]
	folder : str
		prediction的位置
    Returns
    -------
    None.
    """
    # Check Parameter
    assert isinstance(predictions, list), \
        "Parameter predictions must be list, the given type is {}.".format(type(predictions))
    assert len(predictions) > 0, 'Parameter predictions must be a non-emptys list.'
    assert isinstance(predictions[0], tuple), \
        "Elements of list predictions must be a tuple but given {}.".format(type(predictions[0]))

    if folder == None:
        file = "PREDICTIONS.csv"
    else:
        # 检查路径是否存在
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(folder, "PREDICTIONS.csv")

    # 检查文件是否已经存在，如果存在则删除文件
    if os.path.exists(file):
        os.remove(file)
        

    with open(file, mode='w', newline='') as predictions_file:
        # 初始化 CSV 写入器以写入文件
        predictions_writer = csv.writer(predictions_file, delimiter=',')
        # 迭代每个预测
        for prediction in predictions:
            predictions_writer.writerow([prediction[0], prediction[1], prediction[2]])
        # 显示信息保存了多少标签（预测）
        print("{} labels are saved.".format(len(predictions)))

def save_score(f1:float, f1_mult:float, accuracy:float, precision:float, recall:float,folder: str= "") -> None:
    if folder == "":
        file = "SCORE.csv"
    else:
        file = os.path.join(folder, "SCORE.csv")
    # 检查文件是否已经存在，如果存在则删除文件
    if os.path.exists(file):
        os.remove(file)
    # 将参数写入csv文件
    with open(file, mode='w', newline='') as score_file:
        score_writer = csv.writer(score_file, delimiter=',')
        label_list = ["f1", "accuracy","precision","recall","f1_mult"]
        score_list = [f1, accuracy,precision, recall, f1_mult]
        score_writer.writerow(label_list)
        score_writer.writerow(score_list)