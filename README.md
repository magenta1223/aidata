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


<br/><br/>
# Demo files

다른 데모파일도 동일한 형식과 파라미터를 가진다. 따라서 deraining만을 예시로 설명한다. 

## derain_demo.sh

demo shell script for deraining & segmentation.

### Options (Eng)

- —img: single image file path.
- —input_dir: directory containing images for inference
- —Dconfig: Restormer deraining configuration path.
- —Dweighs: pretrained Restormer deraining weights. When you correctly follow the installation guide, all the configurations & weights for Restormer exists in the path.
- —Sconfig: mmsegmentation configuration files. configs/segmentation/{**MODEL_NAME**}_{**DATASET/TASK**}.py. If you use predefined configuration, the pretrained weights for the corresponding model will be automatically downloaded.
- device: devices for this task. If you have GPU, “cuda:0” is recommended. or not, “cpu”
- result_dir: output directory for the results. When the path dose not exist, this directory will be generated.
- out_file: output file name

### 옵션 (Kor)

- —img: 단일 이미지 파일 사용 시, 그 경로
- —input_dir: 추론할 이미지가 담긴 디렉토리.
- —Dconfig: Restormer deraining을 위한 configuration 파일의 경로.
- —Dweighs: deraining Restormer의 사전훈련된 가중치. 설치 가이드를 정상적으로 수행했다면, demo file의 경로에 모든 configuration과 weights가 존재한다.
- —Sconfig: mmsegmentation configuration 파일. configs/segmentation/{**모델이름**}_{**데이터셋 /태스크이름**}.py 형태로 구성된다. 만약, mmsegmentation에서 제공하는 configuration을 사용한다면, 그 configuration에 해당하는 모델의 사전훈련된 가중치가 자동으로 다운로드, 적용된다.
- —Sweight: predefined model을 사용하지 않을 경우 필요한 segmentation model의 가중치의 경로.
- device: 작업에 사용할 장치. GPU가 있다면 “cuda:0”를 추천한다. 없다면 “cpu”를 사용할 것.
- result_dir: 작업 수행 후 출력물을 저장할 디렉토리. 존재하지 않는 경우 자동으로 생성된다.
- out_file: 단일 이미지의 경우 출력물의 이름을 지정할 수 있다.
