
# find Best GaussianMixture Parameter
# data - > [[train, validation], test]

from sklearn import datasets
from sklearn import mixture
import numpy as np
import datasplit
import copy

def get_data():
    fold_num = 3  # 50 data for each folds( 150 / 3 )

    iris = datasets.load_iris()
    X = list(iris.data[:, :])  # 150sampels, 4 features
    y = list(iris.target)  # 3 kinds of target

    # 1-1 : split into 3 folds
    # ( data ) -> (fold1, fold2, fold3 )
    data_folds, target_folds = datasplit.divide_fold(X, y, fold_num)

    # 1-2 : split into train and test for each fold
    # ( fold ) -> ( train, test )
    num_test = int(len(data_folds[0]) / 5)  # 10 test data for each fold
    num_train = int(len(data_folds[0]) - num_test)  # 40 train data for each fold
    data_folds, target_folds = datasplit.get_train_and_test(data_folds, target_folds, fold_num, num_train=num_train,
                                                  num_test=num_test)

    # ( train, test ) -> ( (train1, train2 ... train5) , test )
    num_val = int(len(data_folds[0][0]) / 5)  # 8 val data for each fold
    num_train = int(len(data_folds[0][0]) - num_val)  # 32 val data for each fold
    data_folds, target_folds = datasplit.get_trainfold(data_folds, target_folds, fold_num, num_train=num_train, num_val=num_val)

    return data_folds, target_folds

def get_valdata(data, target, val_fold):
    # divide train data into train, validation

    train_data = copy.deepcopy(data[0])
    train_target = copy.deepcopy(target[0])

    val_data = train_data.pop(val_fold)
    val_target = train_target.pop(val_fold)

    tmp_d, tmp_t = [], []
    for itr in range(len(train_data)):
        tmp_d.extend(train_data[itr])
        tmp_t.extend(train_target[itr])

    train_data = tmp_d
    train_target = tmp_t

    return train_data, train_target, val_data, val_target


def find_proper_parameter(data, target, num_mixture):
    # data : data of folds0 ( [ train, test] )
        # train : [ train1, train2, ...., train5 ]
    # target : target of folds0 ( [ train, test] )
        # train : [ train1, train2, ...., train5 ]

    tot_data = copy.deepcopy(data)
    tot_target = copy.deepcopy(target)

    total_acc = []
    for itr in range(num_mixture):

        accuracy = []
        for i in range(len(data[0])):
            #create Gaussian Mixture model for each condition
            model = mixture.GaussianMixture(n_components=itr+1, tol=1e-3, max_iter=500, random_state=1)
            train_data, train_target, val_data, val_target = get_valdata(tot_data, tot_target, val_fold=int(i))
            model.fit(train_data)
            # find clustered number
            train_predict = model.predict(train_data)

            c_list = []

            # get the predidcted class by train_data
            for c in range(3):
                c_index = list(np.where(np.array(train_target) == c)[0]) #find the data index of each class
                if c_index:
                    p_class = train_predict[c_index[0]]
                    if p_class in c_list:
                        c_list.extend([20])     # if there is no match target, then give trash value
                    else:
                        c_list.extend([p_class])

                if not c_index: #if there is no matched index
                    c_index.extend([20])        # if there is no match target, then give trash value


            #chanege the target class to match trained class
            target = []
            for t in val_target:
                if t == 0: target.append(c_list[0])
                elif t == 1: target.append(c_list[1])
                elif t == 2: target.append(c_list[2])

            predict = model.predict(val_data)
            acc = float( len(list(np.where(np.array(target) == predict)[0])) )
            acc = acc/ 8.

            accuracy.append(acc)
        print('validation accuracy : ', accuracy)
        model_acc = sum(accuracy)/len(accuracy)

        total_acc.append(model_acc)

    print('accuracy of each model done by validation set: ', total_acc)
    max_acc = max(total_acc)
    max_num = total_acc.index(max_acc)

    return max_num

def get_test(data, target, model_num):
    train_data = copy.deepcopy(data[0])
    train_target = copy.deepcopy(target[0])

    test_data = copy.deepcopy(data[1])
    test_target = copy.deepcopy(target[1])

    tmp_d, tmp_t = [], []
    for i in range(len(train_data)):
        tmp_d = tmp_d + train_data[i]
        tmp_t = tmp_t + train_target[i]
    train_data = tmp_d
    train_target = tmp_t

    model = mixture.GaussianMixture(n_components=model_num+1, tol=1e-3, max_iter=500, random_state=1)
    model.fit(train_data)

    # find clustered number
    train_predict = model.predict(train_data)
    c_list = []
    # get the predidcted class by train_data
    for c in range(3):
        c_index = list(np.where(np.array(train_target) == c)[0])  # find the data index of each class
        if c_index:
            p_class = train_predict[c_index[0]]
            if p_class in c_list:
                c_list.extend([20])  # if there is no match target, then give trash value
            else:
                c_list.extend([p_class])

        if not c_index:  # if there is no matched index
            c_index.extend([20])  # if there is no match target, then give trash value

    # chanege the test target class -> match trained class
    target = []
    for t in test_target:
        if t == 0:
            target.append(c_list[0])
        elif t == 1:
            target.append(c_list[1])
        elif t == 2:
            target.append(c_list[2])

    predict = model.predict(test_data)
    acc = float( len(list(np.where(np.array(target) == predict)[0])) )
    acc = acc/ (len(target))

    return acc

if __name__ == '__main__':
    # homework1 :  1~10, possible num of mixtures
    num_mixture = 10

    data_folds, target_folds = get_data()

    # homework2 : make Gaussian mixture and  train
    # 2-1: make Gaussain Mixture
    data = data_folds[0]
    target = target_folds[0]

    model_num = find_proper_parameter(data, target, num_mixture)
    print('selected model num(component number) : ', model_num+1)

    test_acc = get_test(data, target, model_num)
    print('final total accuracy : ', test_acc)
