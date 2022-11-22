python predict.py --model_path "experiments/four_class/CNN/2022_11_20/ECGCNN_S_physio_net_dataset/models/best_model.pt" \
                  --network_config "ECGCNN_S" --fs 500 --use_pretrained \
                  --save_prediction "caizhiwu" --test_data data/test/caizhiwu --file_type "csv"
python predict.py --model_path "experiments/four_class/CNN/2022_11_20/ECGCNN_M_physio_net_dataset/models/best_model.pt" \
                  --network_config "ECGCNN_M" --fs 500 --use_pretrained \
                  --save_prediction "caizhiwu" --test_data data/test/caizhiwu --file_type "csv"
python predict.py --model_path "experiments/four_class/CNN/2022_11_20/ECGCNN_L_physio_net_dataset/models/best_model.pt" \
                  --network_config "ECGCNN_L" --fs 500 --use_pretrained \
                  --save_prediction "caizhiwu" --test_data data/test/caizhiwu --file_type "csv"
python predict.py --model_path "experiments/four_class/CNN/2022_11_20/ECGCNN_XL_physio_net_dataset/models/best_model.pt" \
                  --network_config "ECGCNN_XL" --fs 500 --use_pretrained \
                  --save_prediction "caizhiwu" --test_data data/test/caizhiwu --file_type "csv"
python predict.py --model_path "experiments/four_class/AttNet/2022_11_21/ECGAttNet_S_physio_net_dataset/models/best_model.pt" \
                  --network_config "ECGAttNet_S" --fs 500 --use_pretrained \
                  --save_prediction "caizhiwu" --test_data data/test/caizhiwu --file_type "csv"
python predict.py --model_path "experiments/four_class/AttNet/2022_11_21/ECGAttNet_M_physio_net_dataset/models/best_model.pt" \
                  --network_config "ECGAttNet_M" --fs 500 --use_pretrained \
                  --save_prediction "caizhiwu" --test_data data/test/caizhiwu --file_type "csv"
python predict.py --model_path "experiments/four_class/AttNet/2022_11_21/ECGAttNet_L_physio_net_dataset/models/best_model.pt" \
                  --network_config "ECGAttNet_L" --fs 500 --use_pretrained \
                  --save_prediction "caizhiwu" --test_data data/test/caizhiwu --file_type "csv"
python predict.py --model_path "experiments/four_class/AttNet/2022_11_21/ECGAttNet_XL_physio_net_dataset/models/best_model.pt" \
                  --network_config "ECGAttNet_XL" --fs 500 --use_pretrained \
                  --save_prediction "caizhiwu" --test_data data/test/caizhiwu --file_type "csv"