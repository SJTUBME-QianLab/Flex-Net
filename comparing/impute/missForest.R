## !/user/bin/env
## -*- coding: utf-8 -*-
# install.packages("missForest")
# use mice to impute data

library(missForest)
library(data.table)
Args <- commandArgs()
data_name = Args[3]
lost_name = Args[4]
rand = Args[5]
fold = Args[6]
# data_name = 'arti'
# lost_name = 'L33'
set.seed(2021)

current_path = getwd()
data_path = paste0(strsplit(current_path, split='official_code')[[1]][1], '/official_code/data/')
save_path = paste0(strsplit(current_path, split='official_code')[[1]][1], '/official_code/data/imputed_data/')


drop_allna <- function(train, test){
  tr_nan = which(colSums(is.na(train)) == nrow(train))  # all nan cols (features)
  te_nan = which(colSums(is.na(test)) == nrow(test))
  if (length(tr_nan)==0){
    idx = te_nan
  }else if (length(te_nan)==0){
    idx = tr_nan
  }else{
    idx = c(tr_nan, te_nan)
    idx = idx[!duplicated(idx)]
  }
  if(length(idx) == 0) return(list(train, test))
  tr = train[, -idx]
  te = test[, -idx]
  return(list(tr, te))
}


load_data <- function(rand, fold){
  if (is.element(data_name, c('BC', 'CC', 'CK', 'HC', 'HD', 'HP', 'HS', 'PI'))){
    data_dir = paste0(data_path, '/uci/', data_name, '/divide_', rand)
    data_tr = fread(paste0(data_dir, '/index_data_label_lost_', fold, '_train.csv'))
    data_te = fread(paste0(data_dir, '/index_data_label_lost_', fold, '_test.csv'))
  } else if (is.element(data_name, c('MCAR', 'MAR', 'MNAR'))){
    data_dir = paste0(data_path, '/simu/simu_1100_', data_name, '/divide_', rand, '/', lost_name)  # ���Զ���������V1/V2...
    data_tr = fread(paste0(data_dir, '/index_data_label_lost_', fold, '_train.csv'))
    data_te = fread(paste0(data_dir, '/index_data_label_lost_', fold, '_test.csv'))
  } else if (is.element(data_name, c('nat1', 'nat2'))){
    data_dir = paste0(data_path, '/nat/', data_name, '_0.8_', '/divide_', rand)
    data_tr = fread(paste0(data_dir, '/index_data_label_lost_', fold, '_train.csv'))
    data_te = fread(paste0(data_dir, '/index_data_label_lost_', fold, '_test.csv'))
  } else {
    print('wrong data_name')
  }
  return(list(data_tr, data_te))
}

output = paste0(save_path, '/', data_name, '/missForest/rand', rand, '/', lost_name, '/')
if (!file.exists(output)){
  dir.create(output, recursive = TRUE)
}
xx = load_data(rand=rand, fold=fold)
train_x = data.frame(xx[[1]])
test_x = data.frame(xx[[2]])

xx = drop_allna(train_x, test_x)
train_x = data.frame(xx[[1]])
test_x = data.frame(xx[[2]])

train_tmp = train_x[, 2:(ncol(train_x)-2)]
# impu = complete(mice(train_tmp, seed=2021))
impu <- missForest(train_tmp)
train_impu = cbind(cbind(train_x[, 1], impu[[1]]), train_x[, (ncol(train_x)-1):ncol(train_x)])
write.csv(train_impu, file = paste0(output, '/train', fold, '.csv'), row.names=F)

test_tmp = test_x[, 2:(ncol(test_x)-2)]
# impu = complete(mice(test_tmp, seed=2021))
impu <- missForest(test_tmp)
test_impu = cbind(cbind(test_x[, 1], impu[[1]]), test_x[, (ncol(test_x)-1):ncol(test_x)])
write.csv(test_impu, file = paste0(output, '/test', fold, '.csv'), row.names=F)

