# WSDM2022-XMRec Top1 Solution

[The Cross-Market Recommendation task of WSDM CUP 2022](https://xmrec.github.io/wsdmcup/) is about finding solutions to improve individual recommendation systems in resource-scarce target markets by leveraging data from similar high-resource source markets. Finally, our team OPDAI won the first place with NDCG@10 score of 0.6773 on the leaderboard. The training framework and pipeline are shown in the figure below. And our solution to this task will be detailed in the technical report.


<img src=pipeline.png style=width:666px>


## Steps to Run

### 1. ENV Setup

First, build our docker image

```bash
bash control_docker.sh
```
Second, set the conda env for TF1.11 within the docker image

```bash
# set the tf1.11 env
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && bash ~/miniconda.sh -b
# source ~/.bashrc
cd /root/miniconda3/bin
source activate
conda create -y -n TF111 python=3.6 tensorflow-gpu=1.11 keras=2.2.4
# or use absolute path for conda:
# ~/miniconda3/bin/conda create -y -n TF111 python=3.6 tensorflow-gpu=1.11 keras=2.2.4
# conda activate TF111 
source activate TF111 
cd /home/workspace/src && /root/miniconda3/envs/TF111/bin/pip install -r requirements_tf_4lgcn_v1.txt
```

### 2. Run the script in the docker

#### 2.1 Train with the cached features
```bash
cd /home/workspace/
# train_main
nohup /usr/bin/python train_main.py > train.log 2>&1 &
```
After running the above codes, you can find the submission from the directory: `/home/workspace/OUTPUT/SUB/FINAL`

#### 2.2 Train everything from scratch

If you want to train everything from scratch (It will be quite time-consuming), you could follow the cmds as follows, 

```bash
cd /home/workspace/
nohup bash run_feas.sh > run_feas.log 2>&1 &  
nohup /usr/bin/python train_main.py --use_pretrain 0 --retrain_all_retrieval 1 > train.log 2>&1 &
```
### 3. For new test_run.tsv set inference
```bash
t1_test_run_path=SET_T1_NEW_TEST_RUN_PATH
t2_test_run_path=SET_T2_NEW_TEST_RUN_PATH
nohup /usr/bin/python train_main.py --use_pretrain 0 --retrain_all_retrieval 0 --t1_test_run_path t1_test_run_path --t2_test_run_path t2_test_run_path > train.log 2>&1 &

# nohup /usr/bin/python train_main.py --use_pretrain 0 --retrain_all_retrieval 0 --t1_test_run_path /home/workspace/DATA/t1/test_run.tsv --t2_test_run_path /home/workspace/DATA/t2/test_run.tsv > train.log 2>&1 &
```
If you retrain all retrieval models, you could set `retrain_all_retrieval` as 1 for loading all new trained features from `/home/workplace/OUTPUT/00_NEW`, or you could just leave it as 0 for loading from our pretrained features located at `/home/workplace/OUTPUT/pretrain_features`.

### 4. Pretrained materials

You can start from our pretrained models located at https://drive.google.com/drive/folders/16wBK4ydrt2Bk1qR_rFhZoGkOCy7ey6a4?usp=sharing.


## Contact
If you have any questions, please contact us via zhangqi21@corp.netease.com.

## Reference

* [WSDM 2022 CUP - Cross-Market Recommendation - Starter Kit](https://github.com/hamedrab/wsdm22_cup_xmrec)
* [wsdm-itemcf-baseline](https://aistudio.baidu.com/aistudio/projectdetail/3142643)
* [lightgcn-tf](https://github.com/kuandeng/LightGCN)
* [lightgcn](https://github.com/gusye1234/LightGCN-PyTorch)
* [Solution to the Debiasing Track of KDD CUP 2020 (Team Rush)](https://github.com/xuetf/KDD_CUP_2020_Debiasing_Rush)
* [KDDCUP_2020_Debiasing_1st_Place](https://github.com/aister2020/KDDCUP_2020_Debiasing_1st_Place)
* [GraphEmbedding](https://github.com/shenweichen/GraphEmbedding)
