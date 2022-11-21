# python -W ignore train.py --cuda_devices "0" --epochs 100 --lr 1e-5 --batch_size 32 --physio_net --dataset_path "data/train/" --network_config "ECGCNN_S" \
#                           --load_network "experiments/four_class/CNN/2022_11_20/ECGCNN_S_physio_net_dataset/models/50.pt"

# python -W ignore train.py --cuda_devices "0" --epochs 100 --lr 1e-5 --batch_size 32 --physio_net --dataset_path "data/train/" --network_config "ECGCNN_M" \
#                           --load_network "experiments/four_class/CNN/2022_11_20/ECGCNN_M_physio_net_dataset/models/50.pt"

# python -W ignore train.py --cuda_devices "0" --epochs 100 --lr 1e-5 --batch_size 32 --physio_net --dataset_path "data/train/" --network_config "ECGCNN_L" \
#                           --load_network "experiments/four_class/CNN/2022_11_20/ECGCNN_L_physio_net_dataset/models/50.pt"

# python -W ignore train.py --cuda_devices "0" --epochs 100 --lr 1e-5 --batch_size 32 --physio_net --dataset_path "data/train/" --network_config "ECGCNN_XL" \
#                           --load_network "experiments/four_class/CNN/2022_11_20/ECGCNN_XL_physio_net_dataset/models/50.pt"       

# python -W ignore train.py --cuda_devices "0" --epochs 100 --lr 1e-5 --batch_size 32 --physio_net --dataset_path "data/train/" --network_config "ECGAttNet_S" \
#                           --load_network "experiments/four_class/AttNet/2022_11_21/ECGAttNet_S_physio_net_dataset/models/50.pt"

# python -W ignore train.py --cuda_devices "0" --epochs 100 --lr 1e-5 --batch_size 32 --physio_net --dataset_path "data/train/" --network_config "ECGAttNet_M" \
#                           --load_network "experiments/four_class/AttNet/2022_11_21/ECGAttNet_M_physio_net_dataset/models/50.pt"

# python -W ignore train.py --cuda_devices "0" --epochs 100 --lr 1e-5 --batch_size 32 --physio_net --dataset_path "data/train/" --network_config "ECGAttNet_L" \
#                           --load_network "experiments/four_class/AttNet/2022_11_21/ECGAttNet_L_physio_net_dataset/models/50.pt"

python -W ignore train.py --cuda_devices "0" --epochs 100 --lr 1e-5 --batch_size 32 --physio_net --dataset_path "data/train/" --network_config "ECGAttNet_XL" \
                          --load_network "experiments/four_class/AttNet/2022_11_14/ECGAttNet_XL_physio_net_dataset/models/50.pt"                             