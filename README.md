# Flex-Net

This repository holds the code for the paper

**Flexible Direct Classification for Incomplete Data without Imputation**

All the materials released in this library can **ONLY** be used for **RESEARCH** purposes and not for commercial use.

The authors' institution (Biomedical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University) preserve the copyright and all legal rights of these codes.

# Author List

Xinlu Tang, Rui Guo, Wenli Fu, and Xiaohua Qian

# Abstract

Intelligent approaches for classifying incomplete data hold paramount practical significance due to the common occurrence of missing values. Current classification approaches generally depend on explicit or implicit deletion and imputation. However, they suffer from distorted information, insufficient meaningfulness, and consequently, limited applicable scenarios, particularly in the medical field. To broaden the applicability for the classification of incomplete data, we propose an innovative direct paradigm that neither requires any form of deletion nor imputation. We develop a flexible neural-network-based method, termed Flex-Net, which can flexibility handle samples with various missing patterns (i.e., different sets of available attributes), highlighting its expanded applicable real-world scenarios. Such flexibility is acquired by the experience accumulation across varying scenarios during training, based on a meta-learning framework with an adaptive graph neural network. Extensive experiments illustrate the consistently superior performance of Flex-Net across multiple real-world and simulated datasets, involving various complex practical scenarios characterized by significant ratios of incomplete samples and attributes, as well as diverse and unseen missing patterns. This study suggests that adopting a direct approach for incomplete data classification can fully harness the value of real-world data, prevent inappropriate imputations, and thereby, maximizing the potential of intelligent classification technology in real-world applications.

# Requirements

Our code is mainly based on **Python 3.6** and **PyTorch 1.2.0**. The corresponding environment may be created via conda as follows:

```shell
conda env create -f ./environment/torch1.2.0_env.yaml
conda activate torch1.2.0_env
```

# Raw Data

Raw data contain the following two datasets.

1. TADPOLE
   
   Details of TADPOLE challenge data are available at https://tadpole.grand-challenge.org. The processed data can be downloaded from the official website of ADNI at http://adni.loni.usc.edu/. ADNI requires a user to register and request data with justification of data use.
   
   Details:
   
   - Log in ADNI database
   
   - Access the webpage: https://ida.loni.usc.edu/pages/access/studyData.jsp?categoryId=43&subCategoryId=94
   
   - Click on "Tadpole Challenge Data" button and get a package named "tadpole_challenge_201911210.zip"
   
   - Unzip it and copy the file named `TADPOLE_D1_D2.csv` to `./raw_data/TADPOLE`

2. UCI
   
   - BC -- Breast Cancer
   
   - CC -- Cervical Cancer
   
   - CK -- Chronic Kidney Disease
   
   - HC -- Horse Colic
   
   - HD -- heart-disease
   
   - HP -- Hepatitis
   
   - HS -- hcc-survival
   
   - PI -- pima-indians-diabetes

# Example

Use the first TADPOLE dataset (with natual missing values) as an example to demonstrate the usage of our code.

```shell
conda activate torch1.2.0_env
cd ./Flex-Net/
bash example_nat1.sh
```

# Data processing

1. Naturally incomplete datasets from TADPOLE
   
   ```shell
   # generate data csv for experiments from raw database
   cd ./preprocessing/
   python data_generate_TADPOLE.py --data_root ./../data/ --data_name nat1
   python data_generate_TADPOLE.py --data_root ./../data/ --data_name nat2
   
   # get missing patterns and statistics
   python refine_miss.py --data_root ./../data/ --data_name nat1
   python refine_miss.py --data_root ./../data/ --data_name nat2
   
   # get 1 split for 5-CV
   python data_process_nat.py --data_root ./../data/ --data_name nat1 --rand 1
   python data_process_nat.py --data_root ./../data/ --data_name natme nat2 --job 5
   
   # repeat 10 times (10 splits with 5-CV)
   cd ./reproducibility/
   python data_process_batch.py --data_root ./../data/ --data_name nat1 --job 5
   python data_process_batch.py --data_root ./../data/ --data_name nat2 --job 5
   ```

2. Naturally incomplete datasets from UCI
   
   ```shell
   # generate data csv for experiments from raw database
   cd ./preprocessing/
   python data_generate_UCI.py --data_root ./../data/ --data_name BC
   
   # get 1 split for 5-CV
   python data_process_uci.py --data_root ./../data/ --data_name BC --rand 1
   
   # repeat 10 times (10 splits with 5-CV)
   cd ./reproducibility/
   python data_process_batch.py --data_root ./../data/ --data_name BC--job 5
   ```

3. Simulated datasets from TADPOLE
   
   These datasets have been generated and saved in the package `./data/simu`.
   
   ```shell
   # generate data csv for experiments from raw database
   cd ./preprocessing/
   python data_generate_TADPOLE.py --data_root ./../data/ --data_name simu
   
   # get 1 split for 5-CV
   python data_process_simu.py --data_root ./../data/ --lost_num 0 --rand 1
   python data_process_simu.py --data_root ./../data/ --mode MCAR --lost_num 663 --p_inc 0.05 --rand 1
   
   # repeat 10 times (10 splits with 5-CV)
   cd ./reproducibility/
   python data_process_batch.py --data_root ./../data/ --data_name full --job 5
   python data_process_batch.py --data_root ./../data/ --data_name MCAR --job 5
   ```

# Running code

1. Naturally incomplete datasets from TADPOLE
   
   ```shell
   conda activate torch1.2.0_env
   # 1 fold
   cd ./Flex-Net
   python main_nat.py --data_root ./../data/ --save_dir ./../results/ \
       --data_name nat1 \
       --rand 1 --fold 0 \
       --test_N_way 5 --train_N_way 5 --batch_size 20 --batch_size_test 20 \
       --dec_lr 500 --lr 0.001 --lambda1 0.05
   
   # repeat 10 times (10 splits with 5-CV)
   cd ./../reproducibility/
   python Ours_batch.py --data_root ./../data/ --save_dir ./../results/ --data_name nat1 --job 2
   ```

2. Naturally incomplete datasets from UCI
   
   ```shell
   conda activate torch1.2.0_env
   # 1 fold
   cd ./Flex-Net/
   python main_uci.py --data_root ./../data/ --save_dir ./../results/ \
       --data_name BC \
       --rand 1 --fold 0 \
       --test_N_way 1 --train_N_way 1 --batch_size 20 --batch_size_test 20 \
       --dec_lr 500 --lr 0.0005 --lambda1 0.001
   
   # repeat 10 times (10 splits with 5-CV)
   cd ./reproducibility/
   python Ours_batch.py --data_root ./../data/ --save_dir ./../results/ \
       --data_name BC --job 4
   ```

3. Simulated datasets from TADPOLE
   
   ```shell
   conda activate torch1.2.0_env
   # 1 fold
   cd ./Flex-Net/
   python main_simu.py --data_root ./../data/ --save_dir ./../results/ \
       --data_name MCAR --lost_num 663 --p_inc=0.05 \
       --rand 1 --fold 0 \
       --test_N_way 5 --train_N_way 5 --batch_size 20 --batch_size_test 20 \
       --dec_lr 500 --lr 0.01 --lambda1 0.05
   
   # repeat 10 times (10 splits with 5-CV)
   cd ./reproducibility/
   python Ours_batch.py --data_root ./../data/ --save_dir ./../results/ \
       --data_name MCAR --lost_name L663p0.05 --job 2
   ```

# Comparing methods

## Requirements

We also provide `impute_env.yaml` to perform imputation tools (for `./comparing/impute/impute.py` and `./comparing/impute/missForest.R`) and `tf1.15_env.yaml` to perform GCN classifier (for `./comparing/GCN/main.py` and `./comparing/GCNrisk/main.py`).

## Data processing for deletion-based methods

Delete incomplete features or samples

```shell
cd ./preprocessing/
python drop_incomplete.py --data_root ./../data/ --data_name nat1 --drop Feature --rand 1
# GCNRisk require 4 features: age, gender, education, APOE
python drop_incomplete.py --data_root ./../data/ --data_name nat1 --drop Feature --rand 1 -ex4
```

Batch processing

```shell
cd ./reproducibility/
python data_process_batch.py --data_root ./../data/ --data_name nat1_Feature --job 5
```

## Imputation

Imputed data will be save in the package `./data/imputed_data`.

```shell
conda activate impute_env
# 1 fold
cd ./comparing/impute/
python impute.py --data_name nat1 --rand 1 --fold 0 --fill mean
R --no-save <missForest.R nat1 L999 1 0
python impute.py --data_name BC --rand 1 --fold 0 --fill mean
python impute.py --data_name MCAR --lost_name L663p0.05 --rand 1 --fold 0 --fill mean

# repeat 10 times (10 splits with 5-CV)
cd ./reproducibility/
python impute_batch.py --data_root ./../data/ --save_dir ./../data/imputed_data/ \
    --data_name nat1 --job 5
python impute_batch.py --data_root ./../data/ --save_dir ./../data/imputed_data/ \
    --data_name BC --job 5
python impute_batch.py --data_root ./../data/ --save_dir ./../data/imputed_data/ \
    --data_name MCAR --lost_name L663p0.05 --job 5
```

## Imputation/Deletion + RF/SVM

```shell
conda activate torch1.2.0_env
# 1 fold
cd ./comparing/RFSVM/
python main.py --data_name nat1 --rand 1 --fold 0 --fill mean --classify RF
python main.py --data_name nat1_dropFeature --rand 1 --fold 0 --classify RF
python main.py --data_name BC --rand 1 --fold 0 --fill mean --classify RF
python main.py --data_name MCAR --lost_name L663p0.05 --rand 1 --fold 0 --fill mean --classify RF

# repeat 10 times (10 splits with 5-CV)
cd ./reproducibility/
python RFSVM_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name nat1 --classify RF --job 5
python RFSVM_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name nat1_dropFeature --classify RF --job 5
python RFSVM_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name BC --classify RF --job 5
python RFSVM_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name MCAR --lost_name L663p0.05 --classify RF --job 5
```

## Imputation/Deletion + GCN

```shell
conda activate tf1.15
# 1 fold
cd ./comparing/GCN/
python main.py --data_name nat1 --rand 1 --fold 0 --fill mean \
    --dropout 0.1 --decay 0.00001 --hidden1 64 --lr 0.01 --epochs 400
python main.py --data_name nat1_dropFeature --rand 1 --fold 0 --fill mean \
    --dropout 0.1 --decay 0.00001 --hidden1 64 --lr 0.01 --epochs 400
python main.py --data_name BC --rand 1 --fold 0 --fill mean \
    --dropout 0.2 --decay 0.00001 --hidden1 32 --lr 0.05 --epochs 400
main.py --data_name MCAR --lost_name L663p0.05 --rand 1 --fold 0 --fill mean \
    --dropout 0.1 --decay 0.00001 --hidden1 32 --lr 0.01 --epochs 400

# repeat 10 times (10 splits with 5-CV)
cd ./reproducibility/
python GCN_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name nat1 --job 5
python GCN_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name nat1_dropFeature --job 5
python GCN_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name BC --job 5
python GCN_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name MCAR --lost_name L663p0.05 --job 5
```

## Deletion + GCNrisk

```shell
conda activate tf1.15
# 1 fold
cd ./comparing/GCNrisk/
python main.py --data_name nat1_dropFeature --rand 1 --fold 0 \
    --dropout 0.1 --decay 0.00001 --hidden1 64 --lr 0.01 --epochs 400
# repeat 10 times (10 splits with 5-CV)
cd ./reproducibility/
python GCNrisk_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name nat1_dropFeature --job 5
```

## Deletion + AutoMetric

```shell
conda activate torch1.2.0_env
# 1 fold
cd ./comparing/AutoMetric/
python main.py --data_name nat1_dropFeature --rand 1 --fold 0
# repeat 10 times (10 splits with 5-CV)
cd ./reproducibility/
python AutoMetric_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name nat1_dropFeature --job 5
```

# Figures

- All results are saved in `./plot/all_results.xlsx`

- Fig. 3-5 and Extended Data Fig. 3-4 can be generated by the codes in `./plot/`

- Figures (sub-pannels) will be saved in `./plot/figures`

# References

- [GCN] Kipf, T. N. & Welling, M. Semi-Supervised Classification with Graph Convolutional Networks. Preprint athttps://arxiv.org/abs/1609.02907) (2016).

- [GCNrisk] Parisot, S. et al. Disease prediction using graph convolutional networks: Application to Autism Spectrum Disorder and Alzheimer's disease. *Med Image Anal* **48**, 117-130 (2018).

- [AutoMetric] Song, X., Mao, M. & Qian, X. Auto-metric graph neural network based on a meta-learning strategy for the diagnosis of Alzheimer's disease. *IEEE Journal of Biomedical and Health Informatics* **25**, 3141-3152 (2021).

# Contact

For any question, feel free to contact

> Xinlu Tang : [tangxl20@sjtu.edu.cn](mailto:tangxl20@sjtu.edu.cn)
