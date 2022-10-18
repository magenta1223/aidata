# Ubuntu
## git 설치

	sudo apt-get update
	sudo apt-get install -y git


## Anaconda [설치](https://phoenixnap.com/kb/how-to-install-anaconda-ubuntu-18-04-or-20-04)

먼저 curl을 설치하고, curl을 이용해 anaconda 설치 파일을 다운로드

    apt-get install -y curl

임시파일 저장 위치로 working directory를 이동\
curl을 이용해 해당 위치에 anaconda 설치 파일([다른 버전](https://repo.anaconda.com/archive/))을 다운로드 후, bash로 실행.

    curl -O https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
    bash Anaconda3-2022.05-Linux-x86_64.sh
    source ~/.bashrc


## Restormer & MMSEG 설치
새로운 Anaconda 가상환경 생성 및 package download

    git clone https://github.com/magenta1223/aidata.git
    conda create -n aidata python=3.8
    conda activate aidata
    cd aidata
    sh install.sh

Restormer의 pre-trained model weight는 설치 시 다운로드\
MMSEG의 pre-trained model weight는 predefined configuration from mmseg를 사용해 inference(model 테스트) 시 다운로드 됨. 

<br/><br/>

# Window
## git 설치

    https://git-scm.com/downloads


## Anaconda 설치
    
    https://www.anaconda.com/products/distribution


## Restormer & MMSEG 설치
<img alt="show git-bash logo" src="https://cdn.worldvectorlogo.com/logos/git-bash.svg" width="10"> git bash 열고 아래 코드 실행

    git clone https://github.com/magenta1223/aidata.git
    conda create -n aidata python=3.8
    conda activate aidata
    cd aidata
    sh install.sh

Restormer의 pre-trained model weight는 설치 시 다운로드\
MMSEG의 pre-trained model weight는 predefined configuration from mmseg를 사용해 inference(model 테스트) 시 다운로드 됨.

<br/><br/>

# Quick Start

```
from Restormer.run_Restormer import Restormer

img_list = Restormer(
    model_type='Deraining',
    input_dir="./test_imgs/Deraining/test",
    do_save_results=True,
)



```



<br/><br/>

# Model 사용법
## Model 학습
새로운 dataset으로 학습할 시,\
- Restormer : [Restormer github](https://github.com/swz30/Restormer)에서 설명하는 
방법과 동일 
- MMSEG : 

