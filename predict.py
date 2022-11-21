from datetime import datetime
from typing import List, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch_optimizer
import numpy as np
import os
from tqdm import tqdm
from argparse import ArgumentParser

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from config import *
from util import load_references, save_predictions, save_score,load_data_references
from model import *
from dataset import *
from score import cal_score

def _train(network: nn.Module, two_classes: bool) -> nn.Module:
    """
    Private function which trains the given model
    :param network: (nn.Module) Model to be trained
    :param two_classes: (bool) If true only two classes are utilized
    :return: (nn.Module) Trained model
    """
    # Init data logger
    data_logger = Logger(experiment_path_extension="ECGCNN_XL_predict_training")
    # Init optimizer
    optimizer = torch_optimizer.RAdam(params=network.parameters(), lr=1e-03)
    # Init learning rate schedule
    learning_rate_schedule = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=[1 * 100 // 4, 2 * 100 // 4, 3 * 100 // 4], gamma=0.1)
    # Init datasets
    if two_classes:
        training_split = TRAINING_SPLIT_CHALLANGE_2_CLASSES
        validation_split = VALIDATION_SPLIT_CHALLANGE_2_CLASSES
    else:
        training_split = TRAINING_SPLIT_CHALLANGE
        validation_split = VALIDATION_SPLIT_CHALLANGE
    # Load data
    try:
        ecg_leads, ecg_labels, fs, _ = load_references("data/train/")
    except RuntimeError as exception:
        print("Download the PhysioNet training data or change path. Link is in the repo. Full PhysioNet is used!")
        exit(1904)
    training_dataset = DataLoader(
        PhysioNetDataset(ecg_leads=[ecg_leads[index] for index in training_split],
                         ecg_labels=[ecg_labels[index] for index in training_split], fs=fs,
                        #  augmentation_pipeline=AugmentationPipeline(
                        #      AUGMENTATION_PIPELINE_CONFIG if not two_classes else AUGMENTATION_PIPELINE_CONFIG_2C),
                        augmentation_pipeline=None,
                         two_classes=two_classes),
        batch_size=24, num_workers=20, pin_memory=True, drop_last=False, shuffle=True)
    validation_dataset = DataLoader(
        PhysioNetDataset(ecg_leads=[ecg_leads[index] for index in validation_split],
                         ecg_labels=[ecg_labels[index] for index in validation_split], fs=fs,
                         augmentation_pipeline=None,
                         two_classes=two_classes),
        batch_size=24, num_workers=20, pin_memory=True, drop_last=False, shuffle=False)
    # Init model wrapper
    model_wrapper = ModelWrapper(network=network,
                                 optimizer=optimizer,
                                 loss_function=SoftmaxCrossEntropyLoss(
                                     weight=(1., 1) if two_classes else (0.4, 0.7, 0.9, 0.9)),
                                 training_dataset=training_dataset,
                                 validation_dataset=validation_dataset,
                                 data_logger=data_logger,
                                 learning_rate_schedule=learning_rate_schedule,
                                 device="cuda")
    # Perform training
    model_wrapper.train(epochs=100)
    # Load best model
    network.load_state_dict(torch.load(model_wrapper.data_logger.path_models + "/best_model.pt"))
    return network

@torch.no_grad()
def _predict(network: nn.Module, dataset: DataLoader, ecg_names: List[str],
             two_classes: bool, return_probability: bool,
             device: Union[str, torch.device] = "cpu") -> Union[List[Tuple[str, str]], List[Tuple[str, str, float]],
                                                                List[Tuple[str, str, Dict[str, float]]]]:
    """
    Private function to make predictions
    :param network: (nn.Module) Trained model
    :param dataset: (DataLoader) Dataset to be predicted
    :param ecg_names: (List[str]) Name of each sample
    :param two_classes: (bool) If true only two classes are utilized
    :param return_probability: (bool) If true P(AF) is also returned as part of the result tuple (only for binary case)
    :param device: (Union[str, torch.device]) Device to be utilized
    :return: (Union[List[Tuple[str, str]], List[Tuple[str, str, float]]]) List of tuples including name, prediction and
    probability P(AF) if utilized
    """
    # Init list to store predictions
    predictions: Union[List[Tuple[str, str]], List[Tuple[str, str, float]], List[Tuple[str, Dict[str, float]]]] = []
    # Network to device
    network.to(device)
    # Network into eval mode
    network.eval()
    # Init progress bar
    progress_bar = tqdm(total=len(dataset))
    # Prediction loop
    for name, data in zip(ecg_names, dataset):
        # Update progress bar
        progress_bar.update(n=1)
        # Unpack data
        ecg_lead, spectrogram, _ = data
        # Data to cuda
        ecg_lead = ecg_lead.to(device)
        spectrogram = spectrogram.to(device)
        # Make prediction
        prediction: torch.Tensor = network(ecg_lead, spectrogram)
        # Threshold prediction
        prediction_argmax = prediction.argmax(dim=-1)
        # Construct prediction
        if return_probability:
            if two_classes:
                predictions.append((name, _get_prediction_name(prediction=prediction_argmax, two_classes=two_classes),
                                    prediction[..., -1].item()))
            else:
                predictions.append((name, _get_prediction_name(prediction=prediction_argmax, two_classes=two_classes), dict(zip(["N", "O", "A", "~"], prediction.reshape(-1).tolist()))))
        else:
            predictions.append((name, _get_prediction_name(prediction=prediction_argmax, two_classes=two_classes)))
    # Close progress bar
    progress_bar.close()
    return predictions

def _get_prediction_name(prediction: torch.Tensor, two_classes: bool) -> str:
    """
    Function produces string prediction from raw class prediction
    :param prediction: (torch.Tensor) Prediction of the shape [batch size = 1]
    :param two_classes: (bool) If true two class case is utilized
    :return: (str) String including the class name
    """
    # Check batch size
    assert prediction.shape[0] == 1, "Only a batch size of one is supported."
    # Two class case
    if two_classes:
        if int(prediction.item()) == 0:
            return "N"
        elif int(prediction.item()) == 1:
            return "A"
        else:
            raise RuntimeError("Wrong prediction encountered")
    # Four class case
    if int(prediction.item()) == 0:
        return "N"
    elif int(prediction.item()) == 1:
        return "O"
    elif int(prediction.item()) == 2:
        return "A"
    elif int(prediction.item()) == 3:
        return "~"
    else:
        raise RuntimeError("Wrong prediction encountered")


def predict_my_model(ecg_leads: List[np.ndarray], fs: int, ecg_names: List[str],use_pretrained: bool = False, 
                    return_probability: bool = True, device: Union[str, torch.device] = "cpu", is_binary_classifier:bool=False,
                   model:str = "", network = None) -> Union[List[Tuple[str, str]], List[Tuple[str, str, float]], List[Tuple[str, str, Dict[str, float]]]]:
    # Train model if utilized
    if not use_pretrained:
        # Load weights pre-trained on the Icentia11k dataset
        try:
            state_dict = torch.load(model, map_location=device)
        except FileNotFoundError as _:
            print("State dict not found. Download the state dict of ECG-DualNet XL (Icentia11k). "
                  "Link in README. Put the state dict into the relative directory "
                  "experiments/21_05_2021__12_15_06ECGCNN_XL_icentia11k_dataset/models/")
            exit(1904)
        model_state_dict = network.state_dict()
        state_dict = {key: value for key, value in state_dict.items() if model_state_dict[key].shape == value.shape}
        model_state_dict.update(state_dict)
        network.load_state_dict(model_state_dict)
        # Perform training
        network = _train(network=network, two_classes=is_binary_classifier)
    # Load model
    else:
        if is_binary_classifier:
            try:
                state_dict = torch.load(model, map_location=device)
            except FileNotFoundError as _:
                print("State dict not found. Download the state dict of ECG-DualNet XL (two class, challange). "
                      "Link in README. Put the state dict into the relative directory "
                      "experiments/17_12_2021__03_39_19ECGCNN_XL_physio_net_dataset_challange_two_classes/models/")
                exit(1904)
        else:
            try:
                state_dict = torch.load(model, map_location=device)
            except FileNotFoundError as _:
                print("State dict not found. Download the state dict of ECG-DualNet XL (four class, challange). "
                      "Link in README. Put the state dict into the relative directory,such as "
                      "experiments/25_05_2021__02_02_11ECGCNN_XL_physio_net_dataset_challange/models/")
                exit(1904)
        # Apply state dict
        network.load_state_dict(state_dict)
    # Init dataset for prediction
    dataset = PhysioNetDataset(ecg_leads=ecg_leads, ecg_labels=["A"] * len(ecg_leads), fs=fs,
                               augmentation_pipeline=None, two_classes=is_binary_classifier)
    dataset = DataLoader(dataset=dataset, batch_size=1, num_workers=0, pin_memory=False, drop_last=False, shuffle=False)
    # Make prediction
    return _predict(network=network, dataset=dataset, ecg_names=ecg_names, two_classes=is_binary_classifier,
                    return_probability=return_probability, device=device)

if __name__ == '__main__':
    # Manage command line arguments
    parser = ArgumentParser()
    parser.add_argument("--test_data", default="data/test/physio_net2017", type=str,
                    help="location of REFERENCE.csv")
    parser.add_argument("--model_path", default="experiments/four_class/CNN/ECGCNN_S_physio_net_dataset/models/best_model.pt", type=str,
                    help="location of trained model")
    parser.add_argument("--network_config", default="ECGCNN_M", type=str,
                    choices=["ECGCNN_S", "ECGCNN_M", "ECGCNN_L", "ECGCNN_XL", "ECGAttNet_S", "ECGAttNet_M",
                             "ECGAttNet_L", "ECGAttNet_XL", "ECGAttNet_XXL", "ECGAttNet_130M"],
                    help="Type of network configuration to be utilized.")
    parser.add_argument("--save_prediction", default="results", type=str,
                    help="the path to save PREDICTIONS.csv")
    parser.add_argument("--fs", default=300, type=int,
                    help="sampling frequency")
    parser.add_argument("--use_pretrained", default=True, action="store_true",
                    help="whether to utilize pretrained model")
    parser.add_argument("--two_class", default=False, action="store_false",
                    help="Binary flag. If set two classes are utilized. ")
    parser.add_argument("--devices", default="cuda:0", type=str,
                    help="String of cuda device indexes to be used. Indexes must be separated by a comma.")

    # Get arguments
    args = parser.parse_args()
    
    # Network config and prediction save path 
    if args.network_config == "ECGCNN_S":
        config = ECGCNN_CONFIG_S
        folder = args.save_prediction + "/CNN/" + datetime.now().strftime("%Y_%m_%d") + "/S"
    elif args.network_config == "ECGCNN_M":
        config = ECGCNN_CONFIG_M
        folder = args.save_prediction + "/CNN/" + datetime.now().strftime("%Y_%m_%d") + "/M"  
    elif args.network_config == "ECGCNN_L":
        config = ECGCNN_CONFIG_L
        folder = args.save_prediction + "/CNN/" + datetime.now().strftime("%Y_%m_%d") + "/L"  
    elif args.network_config == "ECGCNN_XL":
        config = ECGCNN_CONFIG_XL
        folder = args.save_prediction + "/CNN/" + datetime.now().strftime("%Y_%m_%d") + "/XL"  
    elif args.network_config == "ECGAttNet_S":
        config = ECGAttNet_CONFIG_S
        folder = args.save_prediction + "/AttNet/" + datetime.now().strftime("%Y_%m_%d") + "/S"
    elif args.network_config == "ECGAttNet_M":
        config = ECGAttNet_CONFIG_M
        folder = args.save_prediction + "/AttNet/" + datetime.now().strftime("%Y_%m_%d") + "/M"
    elif args.network_config == "ECGAttNet_L":
        config = ECGAttNet_CONFIG_L
        folder = args.save_prediction + "/AttNet/" + datetime.now().strftime("%Y_%m_%d") + "/L" 
    elif args.network_config == "ECGAttNet_XL":
        config = ECGAttNet_CONFIG_XL
        folder = args.save_prediction + "/AttNet/" + datetime.now().strftime("%Y_%m_%d") + "/XL"
    elif args.network_config == "ECGAttNet_XXL":
        config = ECGAttNet_CONFIG_XXL
        folder = args.save_prediction + "/AttNet/" + datetime.now().strftime("%Y_%m_%d") + "/XXL" 
    else:
        config = ECGAttNet_CONFIG_130M
        folder = folder = args.save_prediction + "/AttNet/" + datetime.now().strftime("%Y_%m_%d") + "/130M"
    
    if args.two_class:
       config["classes"] = 2
    
    if "CNN" in args.network_config:
        network = ECGCNN(config=config)
    else:
        network = ECGAttNet(config=config)

    # Load Data and Reference
    ecg_leads, _, ecg_names = load_data_references(args.test_data) 

    # Prediction
    predictions = predict_my_model(ecg_leads, args.fs, ecg_names, use_pretrained=args.use_pretrained, model=args.model_path, 
                                    device=args.devices, network=network, is_binary_classifier=args.two_class)
    # Save Prediction
    save_predictions(predictions,folder=folder)                   

    # Get scores and save them
    f1, f1_mult, accuracy, precision, recall = cal_score(prediction_path=folder, reference_path=args.test_data)
    save_score(f1=f1, f1_mult=f1_mult,accuracy=accuracy, precision=precision, recall=recall, folder=folder)