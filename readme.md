# This repository includes LiningNet and serves as a codebase of several plug-and-play modules.
# install  
conda create -n seg2lining python=3.8  
conda activate seg2lining  
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  
cd Seg2Lining  
pip install -r requirements.txt  
pip install open3d  
pip install thop
conda install openpyxl
# usage  
(HPC)sintr -t 1:0:0 --exclusive -A SHEIL-SL3-GPU -p ampere  
conda activate seg2lining  
(HPC)module load cuda/11.8  
(HPC)module load cudnn/8.9_cuda-11.8  
cd Seg2Lining  
python prepare.py  
python train.py  
python test.py  
python test_visualise.py  
python demo.py