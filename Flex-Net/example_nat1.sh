# Description: Example of running Flex-Net on the nat1 dataset
cd ./../preprocessing/ || exit
python data_generate_TADPOLE.py --data_root ./../data/ --data_name nat1

# get missing patterns and statistics
python refine_miss.py --data_root ./../data/ --data_name nat1

# get 1 split for 5-CV
python data_process_nat.py --data_root ./../data/ --data_name nat1 --rand 1

# run Flex-Net
cd ./../Flex-Net/ || exit
conda run -n torch1.2.0 python main_nat.py --data_root ./../data/ --save_dir ./../results/ \
    --data_name nat1 \
    --rand 1 --seed 1 --fold 0 \
    --test_N_way 5 --train_N_way 5 --batch_size 20 --batch_size_test 20 \
    --dec_lr 500 --lr 0.001 --lambda1 0.05

