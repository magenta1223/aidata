pip install mmsegmentation
git clone https://github.com/swz30/Restormer.git # 아래로 들어가서 .git 지워야됨. 
cd Restormer
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
python setup.py develop --no_cuda_ext



curl -O https://storage.googleapis.com/golang/go1.11.1.linux-amd64.tar.gz
mkdir -p ~/installed
tar -C ~/installed -xzf go1.11.1.linux-amd64.tar.gz
mkdir -p ~/go


export GOPATH=$HOME/go
export PATH=$PATH:$HOME/go/bin:$HOME/installed/go/bin


go get github.com/prasmussen/gdrive


cd ..
pip install -r requirements.txt