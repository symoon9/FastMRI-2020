# SNU_challenge: Duya(두야)

by Duya Team (Seoyoon Moon, Hyun Park) 

[2021 SNU FastMRI challenge](https://github.com/LISTatSNU/FastMRI_challenge)

## Environment
### Spec
```plain
 Ubuntu 20.04.2 LTS [ubuntu18.04 in docker container)
 GeForce RTX 3090 3개
 Driver Version: 460.91.03    CUDA Version: 11.1    cuDNN Version: 8
 Docker Engine Version: 20.10.6    NVIDIA Docker: 2.6.0
 pytorch:1.9.0+cu111    torchvision==0.10.0+cu111
```
### Method1. Docker

학습을 위해 Docker image [[nemodleo/pytorch:SNUFastMRI-jupyter](https://hub.docker.com/repository/docker/nemodleo/pytorch)]를 사용합니다.

Duya 팀은 root권한을 줄인 도커 이미지[nemodleo/pytorch:FastMRI-jupyter]를 이용하였습니다. 

[참고: [도커파일](https://github.com/nemodleo/Dockerfile/tree/main/nemodleo)]

```bash
docker build -t nemodleo/pytorch:SNUFastMRI-jupyter .
```
```
docker push nemodleo/pytorch:SNUFastMRI-jupyter
```
```
docker run \
--rm \
-it \
-p 10010:8888 \
--gpus all \
-v {SNU_challenge 폴더}:/SNU_challenge \
--ipc=host \
nemodleo/pytorch:SNUFastMRI-jupyter \
/bin/bash
```

### Method2. Other
도커 환경과 동일한 환경은 다음과 같습니다. 도커 환경 기준으로만 동작함을 확인하였습니다.

자세한 환경은 ```/SNU_challenge/requirements.txt```을 참고하세요.
```
pip install -r requirements.txt
```

#### Issue
torchvision ssim에서 문제 발생 시, 다음을 시도하세요.

``` bash
sudo apt update
sudo apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
```

## Data preparation
### Using only images (our best)
**1. Save .h5 images to .npy**
```bash
# assert your current directory is 'SNU_challenge'
python ./Code/utils/data/save_data_npy.py \
--root_train './Data/train/' \
--root_valid './Data/val/' \
--root_test './Data/Image_Leaderboard/' \
--save_dir './Data/image'
```

### Using k-space
**1. Preprocessing of kspace ( Espirit & PCA )**

kspace을 Espirit & PCA하고난 data(4x384x834)를 "../Data/kspace_processed"에 저장하고 data의 outliers의 위치를 모아 outliers[!].npy file을 생성한다. 
 (단, kspace_processed store dir인 "../Data/kspace_processed"에 다른 .npy file을 두지 마시오.)

시간 단축을 위해 병렬로 계산하기를 추천한다. 병렬처리를 따로 구현하지 않았습니다.
```bash
cd Code/

# train kspace data
python preprocessing_kspace.py  -l train -d ../Data/kspace -s ../Data/kspace_processed -n -1 -o y

# test(Leaderboard) kspace data
python preprocessing_kspace.py -l test -d ../Data/kspace_Leaderboard -s ../Data/kspace_processed -n -1  -o y
```

```python preprocessing_kspace.py --dir=DIRECTORY_OF_KSPACE --store-dir=STORE_DIRETORY_OF_KSPACE_PROCESSED --num-target=LIST_OF_TARGET_KSPACE_NUMBERS(ALL_IS_-1) --train-or-test=TRAIN_OR_TEST --outliers-store OUTLIERS_STORE[y/n] --esp-kernel=ESPIRIT_KERNEL_SIZE --esp-region=ESPIRIT_REGION_SIZE --esp-t=ESPIRIT_T_FACTOR --esp-c=ESPIRIT_C_FACTOR``` 을 참고하세요.


**2. Merge outliers files**

1과정에서 생성된 outliers[!].npy들을 하나의 outliers.npy파일로 만든다. 분산처리를 하지 않았다면 두 개이하의 outliers[!].npy을 하나의 outliers.npy파일로 병합합니다.
```bash
cd Code/
python outliers.py -d ../Data/kspace_processed
```

## Training & Make reconstructed files
### Using only images (our best)
1. Modify the `run_script.sh` file
- Our model size is up to 14GB. For data parallelization, set `--gpu '0,1'`
- set directories:
    - `--data_path_train`: Data path to train npy files
    - `--data_path_val`: Data path to validation npy files
    - `--data_path_test`: Data path to test npy files (leader board)
    - `--data_save_test`: Directory to save reconstructed data
```bash
cd Code/
bash run_script.sh
```
### Using k-space
```bash
cd Code/
bash run_script_kspace.sh
```
## Evaluation
```bash
# assert your current directory is 'SNU_challenge'
python ./Code/leaderboard_eval.py
```
## Contact
Please contact us if any question or suggestion

Moon Seoyoon [mrn538@snu.ac.kr](mailto:mrn538@snu.ac.kr), Park Hyun [nemod.leo@snu.ac.kr](mailto:nemod.leo@snu.ac.kr)