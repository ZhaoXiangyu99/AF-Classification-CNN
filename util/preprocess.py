import torch
from torch_ecg.cfg import CFG
from torch_ecg._preprocessors import PreprocManager
import numpy as np

# preprocess ecg signal
def preprocess_ecg(ecg_signal: np.ndarray, fs:int = 500) -> np.ndarray:
    '''
    Parameters
    ----------
    ecg_signal: np.ndarray
        raw ecg signal to be preprocessed
    fs: int,default value is 500
        sampling frequency of raw ecg signal 

    Returns
    -------
    pp_ecg_signal: np.ndarray
        preprocessed ecg signal 
    '''
    config = CFG(
        random=False,
        resample={"fs": 300},
        bandpass={"filter_type": "butter", "low_cut": 0.5, "high_cut": 1000},
        normalize={"method": "z-score"},
    )
    ppm = PreprocManager.from_config(config)
    ecg_signal = torch.from_numpy(ecg_signal)
    ecg_signal= ppm(ecg_signal, fs=fs)
    return ecg_signal[0]