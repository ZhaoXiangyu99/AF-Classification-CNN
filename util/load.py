from typing import List, Tuple
import csv
import scipy.io as sio
import numpy as np
import os
import pandas as pd
from .preprocess import preprocess_ecg

def load_references(folder: str) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Parameters
    ----------
    folder : str, optional
        训练数据的位置。默认值'../training'.
    Returns
    -------
    ecg_leads : List[np.ndarray]
        心电图信号.
    ecg_labels : List[str]
        ECG信号的label，包括: 'N','A','O','~'
    fs : int
        采样频率.
    ecg_names : List[str]
        加载文件的名称
    """
    # Check Parameter
    assert isinstance(folder, str), "Parameter folder must be string".format(type(folder))
    assert os.path.exists(folder), 'Parameter folder  doesn\'t exist!'

    # Initialize ecg_leads,ecg_labels,ecg_names
    ecg_leads: List[np.ndarray] = []
    ecg_labels: List[str] = []
    ecg_names: List[str] = []

    # load REFERENCE.csv
    with open(os.path.join(folder, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # 遍历每一行
        for row in csv_reader:
            # 加载带有 ECG 导联和标签的 MatLab 文件
            data = sio.loadmat(os.path.join(folder, row[0] + '.mat'))
            ecg_leads.append(data['val'][0])   # ecg信号值
            ecg_labels.append(row[1])          # 标签,即所属类别
            ecg_names.append(row[0])           # ecg文件名
    # 显示加载了多少数据
    print("{} items of data are loaded.".format(len(ecg_leads)))
    return ecg_leads, ecg_labels, ecg_names

# load data and reference
def load_data_references(folder: str, file_type: str = "mat") -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Parameters
    ----------
    folder : str, optional
        训练数据的位置。默认值'../training'.
    Returns
    -------
    ecg_leads : List[np.ndarray]
        心电图信号.
    ecg_labels : List[str]
        ECG信号的label，包括: 'N','A','O','~'
    fs : int
        采样频率.
    ecg_names : List[str]
        加载文件的名称
    """
    # Check Parameter
    assert isinstance(folder, str), "Parameter folder must be string".format(type(folder))
    assert os.path.exists(folder), 'Parameter folder  doesn\'t exist!'

    # Initialize ecg_leads,ecg_labels,ecg_names
    ecg_leads: List[np.ndarray] = []
    ecg_labels: List[str] = []
    ecg_names: List[str] = []

    # load REFERENCE.csv
    with open(os.path.join(folder, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # 遍历每一行
        for row in csv_reader:
            ecg_data = load_data(folder=folder, file_name=row[0], file_type=file_type)
            pp_ecg_data = preprocess_ecg(ecg_signal=ecg_data, fs=500)
            ecg_leads.append(pp_ecg_data)     # ecg信号值
            ecg_labels.append(row[1])      # 标签,即所属类别
            ecg_names.append(row[0])       # ecg文件名
    # 显示加载了多少数据
    print("加载了{}条数据.".format(len(ecg_leads)))
    return ecg_leads, ecg_labels, ecg_names

# load training data or testing data
def load_data(folder: str = "", file_name: str = "",file_type: str="mat") -> np.ndarray :
    file_type_list = ["mat", "csv", "txt", "xlsx", "xls"]
    assert file_type in file_type_list, \
        "suffix of data files must be in [\"mat\", \"csv\", \"txt\", \"xlsx\", \"xls\"]"
    if file_type == "mat":
        ecg_data = sio.loadmat(os.path.join(folder, file_name + '.mat'))['val'][0]
    elif file_type == "csv":
        df = pd.read_csv(os.path.join(folder, file_name + ".csv"))
        ecg_data = df.to_numpy()[:, 0]
    elif file_type == "txt":
        pass
    elif file_type == "xls":
        pass
    else:
        pass
    return ecg_data
