#ID : 2018221270
#NAME : jonghoon park
# divide dataset into given numbers of folds
# some of folds are for training, some are validation, and test

from sklearn import datasets
from sklearn.utils import shuffle

import numpy as np

def divide_fold(data, target, fold_num):
    # get total_data and divide into 5 folds
    # data : list , total data list
    # target : list, total target of data
    # fold_num : number of folds

    # shuffle the dataset not disturbing the mapping
    data, target = shuffle(data, target, random_state=0)

    data_folds, target_folds = [], []
    data_num = int(len(data)/fold_num)
    for i in range(fold_num):
        start = data_num * i
        tmp_d = data[start : start + data_num]
        tmp_t = target[start : start + data_num]
        data_folds.append(tmp_d)
        target_folds.append(tmp_t)
    return data_folds, target_folds

def get_train_and_test(data, target, fold_num, num_train, num_test):
    # data : list
        #folds -> ( data )
    # target : list
        #folds -> ( data )
    split_data = []     # ( train, test )
    split_target = []   # ( train, test )

    num_data = len(data[0])
    for i in range(fold_num):

        # [ train_data, test_data ] for each fold
        tmp_d = [ data[i][0:num_train] , data[i][num_train : num_train + num_test]]
        tmp_t = [ target[i][0:num_train], target[i][num_train : num_train + num_test]]

        split_data.append(tmp_d)
        split_target.append(tmp_t)

    return split_data, split_target

def get_trainfold(data, target, fold_num, num_train, num_val):
    # data : list ( train, test )
        #train, test : list
    # target : list ( train, test )
        #train, test : list

    split_data = []     # ( train, val, test )
    split_target = []   # ( train, val, test )

    for i in range(fold_num):
        tmp_train_d = []
        tmp_target_d = []
        for j in range(5):
            start = j * num_val
            tmp_train_d.append(data[i][0][start : start+num_val])
            tmp_target_d.append(target[i][0][start : start+num_val])

        tmp_d = [tmp_train_d, data[i][1]]
        tmp_t = [tmp_target_d, target[i][1]]

        split_data.append(tmp_d)
        split_target.append(tmp_t)

    return split_data, split_target


if __name__ == '__main__':

    fold_num = 3    #50 data for each folds( 150 / 3 )

    iris = datasets.load_iris()
    X = list(iris.data[:,:])      # 150sampels, 4 features
    y = list(iris.target)         # 3 kinds of target

    print('-----------homwork1')
    # homework 1
    # 1-1 : split into 3 folds
    # ( data ) -> (fold1, fold2, fold3 )
    data_folds, target_folds = divide_fold(X, y, fold_num)
    print('splitted data folds number : ', len(data_folds))

    # 1-2 : split into train and test for each fold
    # ( fold ) -> ( train, test )
    num_test = int(len(data_folds[0]) / 5)  # 10 test data for each fold
    num_train = int(len(data_folds[0]) - num_test)  # 40 train data for each fold
    data_folds, target_folds = get_train_and_test(data_folds, target_folds, fold_num, num_train = num_train, num_test= num_test)
    print('number of train data for each fold : ', len(data_folds[0][0]))
    print('number of test data for each fold : ', len(data_folds[0][1]))

    print('\n--------------homeworkd2')
    # homework 2
    # 2-1 : split train into train and validation for each fold
    # ( train, test ) -> ( (train1, train2 ... train5) , test )
    num_val = int(len(data_folds[0][0]) / 5)    # 8 val data for each fold
    num_train = int(len(data_folds[0][0]) - num_val)    # 32 val data for each fold
    data_folds, target_folds = get_trainfold(data_folds, target_folds, fold_num, num_train=num_train, num_val=num_val)

    print('train)number of fold for each fold : ', len(data_folds[0][0]))
    print('train)number of data for each train fold : ', len(data_folds[0][0][0]))
    print('train)number of total data : ', len(data_folds[0][0]) * len(data_folds[0][0][0]))

    print('test)number of data for each fold : ', len(data_folds[0][1]))
