#!/bin/bash
#SBATCH -N 1 #1 node
#SBATCH --time=3-00:00:00
#SBATCH --job-name=crop_classification_unet
#SBATCH --error=%j.err_
#SBATCH --output=$j.out_
#SBATCH --nodelist radagast
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB

module load python
module load cuda/11.1
eval "$(conda shell.bash hook)"


source /opt/miniconda3/bin/activate koala_yolo_3.9
nvidia-smi
nvcc --version
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge


python -m torch.distributed.run --nproc_per_node 4 train.py --img 640 --batch 16 --epochs 3 --data data/coco.yaml --weights yolov5s.pt --device 0,1,3,4