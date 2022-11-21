import pandas as pd
import csv
import os

def cal_score(prediction_path : str = "", reference_path : str = ""):
    prediction_path = prediction_path + "/PREDICTIONS.csv"
    reference_path = reference_path + "/REFERENCE.csv"
    df_pred = pd.read_csv(prediction_path, header=None)  # 预测的csv文件
    df_gt = pd.read_csv(reference_path, header=None)  # 真实的Reference文件

    N_files = df_gt.shape[0]  # Anzahl an Datenpunkten

    ## normalize F1-Score
    TP = 0  # True Positive
    TN = 0  # True Negative
    FP = 0  # False Positive
    FN = 0  # False Negative

    ## Multi-Class-F1
    '''
        N : Normal
        A : Atrial Fibrillation
        P : Noisy
        O : Other rhythm
    '''
    Nn = 0  # 真实是N,预测为N
    Na = 0  # 真实是N,预测为A
    No = 0  # 真实是N,预测为O
    Np = 0  # 真实是N,预测为P
    An = 0  # 真实是A,预测为N
    Aa = 0  # 真实是A,预测为A
    Ao = 0  # 真实是A,预测为O
    Ap = 0  # 真实是A,预测为P
    On = 0  # 真实是O,预测为N
    Oa = 0  # 真实是O,预测为A
    Oo = 0  # 真实是O,预测为O
    Op = 0  # 真实是O,预测为P
    Pn = 0  # 真实是P,预测为N
    Pa = 0  # 真实是P,预测为A
    Po = 0  # 真实是P,预测为O
    Pp = 0  # 真实是P,预测为P

    for i in range(N_files):
        gt_name = df_gt[0][i]
        gt_class = df_gt[1][i]

        pred_indx = df_pred[df_pred[0] == gt_name].index.values

        if not pred_indx.size:
            print("Prediction for " + gt_name + " fehlt, nehme \"normal\" an.")
            pred_class = "N"
        else:
            pred_indx = pred_indx[0]
            pred_class = df_pred[1][pred_indx]

        if gt_class == "A" and pred_class == "A":
            TP = TP + 1
        if gt_class == "N" and pred_class == "N":
            TN = TN + 1
        if gt_class == "N" and pred_class == "A":
            FP = FP + 1
        if gt_class == "A" and pred_class == "N":
            FN = FN + 1

        if gt_class == "N":
            if pred_class == "N":
                Nn = Nn + 1
            if pred_class == "A":
                Na = Na + 1
            if pred_class == "O":
                No = No + 1
            if pred_class == "~":
                Np = Np + 1

        if gt_class == "A":
            if pred_class == "N":
                An = An + 1
            if pred_class == "A":
                Aa = Aa + 1
            if pred_class == "O":
                Ao = Ao + 1
            if pred_class == "~":
                Ap = Ap + 1

        if gt_class == "O":
            if pred_class == "N":
                On = On + 1
            if pred_class == "A":
                Oa = Oa + 1
            if pred_class == "O":
                Oo = Oo + 1
            if pred_class == "~":
                Op = Op + 1

        if gt_class == "~":
            if pred_class == "N":
                Pn = Pn + 1
            if pred_class == "A":
                Pa = Pa + 1
            if pred_class == "O":
                Po = Po + 1
            if pred_class == "~":
                Pp = Pp + 1

    sum_N = Nn + Na + No + Np
    sum_A = An + Aa + Ao + Ap
    sum_O = On + Oa + Oo + Op
    sum_P = Pn + Pa + Po + Pp

    sum_n = Nn + An + On + Pn
    sum_a = Na + Aa + Oa + Pa
    sum_o = No + Ao + Oo + Po
    sum_p = Np + Ap + Op + Pp

    F1 = TP / (TP + 1 / 2 * (FP + FN))
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)

    F1_mult = 0
    n_f1_mult = 0

    if (sum_N + sum_n) != 0:
        F1_mult += 2 * Nn / (sum_N + sum_n)
        n_f1_mult += 1

    if (sum_A + sum_a) != 0:
        F1_mult += 2 * Aa / (sum_A + sum_a)
        n_f1_mult += 1

    if (sum_O + sum_o) != 0:
        F1_mult += 2 * Oo / (sum_O + sum_o)
        n_f1_mult += 1

    if (sum_P + sum_p) != 0:
        F1_mult += 2 * Pp / (sum_P + sum_p)
        n_f1_mult += 1

    F1_mult = F1_mult / n_f1_mult

    return F1, F1_mult, Accuracy, Precision, Recall





