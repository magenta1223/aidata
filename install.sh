# Create Virtual Env
# conda create -n aidata python==3.8
# conda activate aidata

pip install wget

# Inference code
git clone https://github.com/magenta1223/aidata.git
cd aidata

# Restormer
git clone https://github.com/swz30/Restormer.git
cd Restormer

# RTX 30 Series
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# or lower 
# conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch

# requirements for restormer
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips

# Restormer setup
python setup.py develop --no_cuda_ext
rm -rf .git 
cd ..


# mmsegmentation & mmcv
#pip install mmsegmentation
#pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
pip install -U openmim
mim install mmcv-full

git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
rm -rf .git 
cd ..

# configurations 

mkdir configs
cp -r Restormer/Defocus_Deblurring/Options/ configs/defocus_deblur/
cp -r Restormer/Denoising/Options/ configs/denoise/
cp -r Restormer/Deraining/Options/ configs/derain/
cp -r Restormer/Motion_Deblurring/Options/ configs/motion_deblur/
cp -r mmsegmentation/configs/ configs/segmentation


# Restormer weights
## Derain weights
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uuejKpyo0G_5M4DAO2J9_Dijy550tjc5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uuejKpyo0G_5M4DAO2J9_Dijy550tjc5" -O backup.zip && rm -rf /tmp/cookies.txt
mv backup.zip configs/derain/derain.pth
rm -f backup.zip

## Denoise weights
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pwcOhDS5Erzk8yfAbu7pXTud606SB4-L' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1pwcOhDS5Erzk8yfAbu7pXTud606SB4-L" -O backup.zip && rm -rf /tmp/cookies.txt
mv backup.zip configs/denoise/deblur.pth
rm -f backup.zip

## Motion Deblur weights
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pwcOhDS5Erzk8yfAbu7pXTud606SB4-L' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1pwcOhDS5Erzk8yfAbu7pXTud606SB4-L" -O backup.zip && rm -rf /tmp/cookies.txt
mv backup.zip configs/motion_deblur/motion_deblur.pth
rm -f backup.zip

## Defocus & Deblur weights
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10v8BH3Gktl34TYzPy0x-pAKoRSYKnNZp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10v8BH3Gktl34TYzPy0x-pAKoRSYKnNZp" -O backup.zip && rm -rf /tmp/cookies.txt
mv backup.zip configs/defocus_deblur/single_image_defocus_deblurring.pth
rm -f backup.ziphttps://drive.google.com/file/d/167enijHIBa1axZRaRjkk_U6kLKm40Z43/view?usp=sharing

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=167enijHIBa1axZRaRjkk_U6kLKm40Z43' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=167enijHIBa1axZRaRjkk_U6kLKm40Z43" -O backup.zip && rm -rf /tmp/cookies.txt
mv backup.zip configs/defocus_deblur/dual_pixel_defocus_deblurring.pth
rm -f backup.zip