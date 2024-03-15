
#python -c "import torch; print(torch.__version__)"

######################## 1. TADPOLE application ########################

# generate data csv for experiments from raw database
cd ./../preprocessing/ || exit
python data_generate_TADPOLE.py --data_root ./../data/ --data_name nat1
python data_generate_TADPOLE.py --data_root ./../data/ --data_name nat2

# get missing patterns and statistics
python refine_miss.py --data_root ./../data/ --data_name nat1
python refine_miss.py --data_root ./../data/ --data_name nat2

# get 1 split for 5-CV
python data_process_nat.py --data_root ./../data/ --data_name nat1 --rand 1
python data_process_nat.py --data_root ./../data/ --data_name nat2 --rand 1

# repeat 10 times (10 splits with 5-CV)
cd ./../reproducibility/ || exit
python data_process_batch.py --data_root ./../data/ --data_name nat1 --job 5
python data_process_batch.py --data_root ./../data/ --data_name nat2 --job 5

## deletion-based method, drop incomplete Features or Samples
#cd ./../preprocessing/ || exit
#python drop_incomplete.py --data_root ./../data/ --data_name nat1 --drop Feature --rand 1
## GCNRisk require 4 features: age, gender, education, APOE
#python drop_incomplete.py --data_root ./../data/ --data_name nat1 --drop Feature --rand 1 -ex4
#
# repeat 10 times (10 splits with 5-CV)
#cd ./../reproducibility/ || exit
#python data_process_batch.py --data_root ./../data/ --data_name nat1_Feature --job 5
#python data_process_batch.py --data_root ./../data/ --data_name nat1_Sample --job 5
#python data_process_batch.py --data_root ./../data/ --data_name nat2_Feature --job 5

## Ours --------------------------------
# 1 fold
#cd ./../Flex-Net/ || exit
#conda run -n torch1.2.0 python main_nat.py --data_root ./../data/ --save_dir ./../results/ \
#    --data_name nat1 \
#    --rand 1 --seed 1 --fold 0 \
#    --test_N_way 5 --train_N_way 5 --batch_size 20 --batch_size_test 20 \
#    --dec_lr 500 --lr 0.001 --lambda1 0.05
#conda run -n torch1.2.0 python main_nat.py --data_root ./../data/ --save_dir ./../results/ \
#    --data_name nat2 \
#    --rand 2 --seed 2 --fold 0 \
#    --test_N_way 5 --train_N_way 5 --batch_size 20 --batch_size_test 20 \
#    --dec_lr 500 --lr 0.001 --lambda1 0.05

## repeat 10 times (10 splits with 5-CV)
#cd ./../reproducibility/ || exit
#conda run -n torch1.2.0 python Ours_batch.py --data_root ./../data/ --save_dir ./../results/ --data_name nat1 --job 2

## impute --------------------------------
## 1 fold
#cd ./../comparing/impute/ || exit
#conda run -n impute python impute.py --data_name nat1 --rand 1 --fold 0 --fill mean
#conda run -n impute R --no-save <missForest.R nat1 L999 1 0
## repeat 10 times (10 splits with 5-CV)
#cd ./../reproducibility/ || exit
#conda run -n impute python impute_batch.py --data_root ./../data/ --save_dir ./../data/imputed_data/ \
#    --data_name nat1 --job 5

## RF & SVM --------------------------------
## 1 fold
#cd ./../comparing/RFSVM/ || exit
#conda run -n torch1.2.0 python main.py --data_name nat1 --rand 1 --fold 0 --fill mean --classify RF
#conda run -n torch1.2.0 python main.py --data_name nat1_dropFeature --rand 1 --fold 0 --classify RF
## repeat 10 times (10 splits with 5-CV)
#cd ./../reproducibility/ || exit
#conda run -n torch1.2.0 python RFSVM_batch.py --data_root ./../data/ --save_dir ./../results/ \
#    --data_name nat1 --classify RF --job 5
#conda run -n torch1.2.0 python RFSVM_batch.py --data_root ./../data/ --save_dir ./../results/ \
#    --data_name nat1_dropFeature --classify RF --job 5

## GCN --------------------------------
## 1 fold
#cd ./../comparing/GCN/ || exit
#conda run -n tf1.15 python main.py --data_name nat1 --rand 1 --fold 0 --fill mean \
#    --dropout 0.1 --decay 0.00001 --hidden1 64 --lr 0.01 --epochs 400
#conda run -n tf1.15 python main.py --data_name nat1_dropFeature --rand 1 --fold 0 --fill mean \
#    --dropout 0.1 --decay 0.00001 --hidden1 64 --lr 0.01 --epochs 400
## repeat 10 times (10 splits with 5-CV)
#cd ./../reproducibility/ || exit
#conda run -n tf1.15 python GCN_batch.py --data_root ./../data/ --save_dir ./../results/ \
#    --data_name nat1 --job 5
#conda run -n tf1.15 python GCN_batch.py --data_root ./../data/ --save_dir ./../results/ \
#    --data_name nat1_dropFeature --job 5

# GCNRisk --------------------------------
# 1 fold
cd ./../comparing/GCNrisk/ || exit
conda run -n tf1.15 python main.py --data_name nat1_dropFeature --rand 1 --fold 0 \
    --dropout 0.1 --decay 0.00001 --hidden1 64 --lr 0.01 --epochs 400 --seed 231
# repeat 10 times (10 splits with 5-CV)
cd ./../reproducibility/ || exit
conda run -n tf1.15 python GCNrisk_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name nat1_dropFeature --job 5

# AutoMetric --------------------------------
# 1 fold
cd ./../comparing/AutoMetric/ || exit
conda run -n torch1.2.0 python main.py --data_name nat1_dropFeature --rand 1 --fold 0 --seed 2019
# repeat 10 times (10 splits with 5-CV)
cd ./../reproducibility/ || exit
conda run -n torch1.2.0 python AutoMetric_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name nat1_dropFeature --job 5


####################### 2. UCI ########################

# generate data csv for experiments from raw database
cd ./../preprocessing/ || exit
python data_generate_UCI.py --data_root ./../data/ --data_name BC

# get 1 split for 5-CV
python data_process_uci.py --data_root ./../data/ --data_name BC --rand 1

# repeat 10 times (10 splits with 5-CV)
cd ./../reproducibility/ || exit
python data_process_batch.py --data_root ./../data/ --data_name BC --job 5

## Ours --------------------------------
## 1 fold
#cd ./../Flex-Net/ || exit
#conda run -n torch1.2.0 python main_uci.py --data_root ./../data/ --save_dir ./../results/ \
#    --data_name BC \
#    --rand 1 --seed 1 --fold 0 \
#    --test_N_way 1 --train_N_way 1 --batch_size 20 --batch_size_test 20 \
#    --dec_lr 500 --lr 0.0005 --lambda1 0.001
#
## repeat 10 times (10 splits with 5-CV)
#cd ./../reproducibility/ || exit
#conda run -n torch1.2.0 python Ours_batch.py --data_root ./../data/ --save_dir ./../results/ \
#    --data_name BC --job 4


# impute --------------------------------
# 1 fold
cd ./../comparing/impute/ || exit
conda run -n impute python impute.py --data_name BC --rand 1 --fold 0 --fill mean

# repeat 10 times (10 splits with 5-CV)
cd ./../reproducibility/ || exit
conda run -n impute python impute_batch.py --data_root ./../data/ --save_dir ./../data/imputed_data/ \
    --data_name BC --job 5

## RF & SVM --------------------------------
## 1 fold
cd ./../comparing/RFSVM/ || exit
conda run -n torch1.2.0 python main.py --data_name BC --rand 1 --fold 0 --fill mean --classify RF
# repeat 10 times (10 splits with 5-CV)
cd ./../reproducibility/ || exit
conda run -n torch1.2.0 python RFSVM_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name BC --classify RF --job 5

# GCN --------------------------------
# 1 fold
cd ./../comparing/GCN/ || exit
conda run -n tf1.15 python main.py --data_name BC --rand 1 --fold 0 --fill mean \
    --dropout 0.2 --decay 0.00001 --hidden1 32 --lr 0.05 --epochs 400 --seed 2
# repeat 10 times (10 splits with 5-CV)
cd ./../reproducibility/ || exit
conda run -n tf1.15 python GCN_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name BC --job 5


####################### 3. TADPOLE simulation ########################

# generate data csv for experiments from raw database
cd ./../preprocessing/ || exit
python data_generate_TADPOLE.py --data_root ./../data/ --data_name simu

# get 1 split for 5-CV
python data_process_simu.py --data_root ./../data/ --lost_num 0 --rand 1
python data_process_simu.py --data_root ./../data/ --mode MCAR --lost_num 663 --p_inc 0.05 --rand 1

# repeat 10 times (10 splits with 5-CV)
cd ./../reproducibility/ || exit
python data_process_batch.py --data_root ./../data/ --data_name full --job 5
python data_process_batch.py --data_root ./../data/ --data_name MCAR --job 5
python data_process_batch.py --data_root ./../data/ --data_name MAR --job 5
python data_process_batch.py --data_root ./../data/ --data_name MNAR --job 5

# Ours
# 1 fold
cd ./../Flex-Net/ || exit
conda run -n torch1.2.0 python main_simu.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name MCAR --lost_num 663 --p_inc=0.05 \
    --rand 1 --seed 1 --fold 0 \
    --test_N_way 5 --train_N_way 5 --batch_size 20 --batch_size_test 20 \
    --dec_lr 500 --lr 0.01 --lambda1 0.05 --seed 4377

# repeat 10 times (10 splits with 5-CV)
cd ./../reproducibility/ || exit
conda run -n torch1.2.0 python Ours_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name MCAR --lost_name L663p0.05 --job 2


# impute
# 1 fold
cd ./../comparing/impute/ || exit
conda run -n impute python impute.py --data_name MCAR --lost_name L663p0.05 --rand 1 --fold 0 --fill mean

# repeat 10 times (10 splits with 5-CV)
cd ./../reproducibility/ || exit
conda run -n impute python impute_batch.py --data_root ./../data/ --save_dir ./../data/imputed_data/ \
    --data_name MCAR --lost_name L663p0.05 --job 5
conda run -n impute python impute_batch.py --data_root ./../data/ --save_dir ./../data/imputed_data/ \
    --data_name MAR --lost_name L331p0.8 --job 5

## RF & SVM --------------------------------
## 1 fold
cd ./../comparing/RFSVM/ || exit
conda run -n torch1.2.0 python main.py --data_name MCAR --lost_name L663p0.05 --rand 1 --fold 0 --fill mean --classify RF
conda run -n torch1.2.0 python main.py --data_name MAR --lost_name L331p0.8 --rand 1 --fold 0 --fill mean --classify RF --seed 2032
# repeat 10 times (10 splits with 5-CV)
cd ./../reproducibility/ || exit
conda run -n torch1.2.0 python RFSVM_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name MAR --lost_name L331p0.8 --classify RF --job 5

# GCN --------------------------------
# 1 fold
cd ./../comparing/GCN/ || exit
conda run -n tf1.15 python main.py --data_name MCAR --lost_name L663p0.05 --rand 1 --fold 0 --fill mean \
    --dropout 0.1 --decay 0.00001 --hidden1 32 --lr 0.01 --epochs 400 --seed 231
# repeat 10 times (10 splits with 5-CV)
cd ./../reproducibility/ || exit
conda run -n tf1.15 python GCN_batch.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name MCAR --lost_name L663p0.05 --job 5
