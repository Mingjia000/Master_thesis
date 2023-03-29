import PL_activation
import pandas as pd
import par
import timeit
import numpy as np

start = timeit.default_timer()

train_ll_l=[]
train_acc_l=[]
train_num_l=[]
test_ll_l=[]
test_acc_l=[]
test_num_l=[]
train_ll=0
train_acc=0
train_num=0
test_ll=0
test_acc=0
test_num=0
acc=np.zeros((10,3))
for model in range (3):
    for i in range(10):
        accuracy, betas = PL_activation.main(model)
        acc[i,model] = accuracy
df_acc = pd.DataFrame(acc,columns=['BL','1-NN','5-NN'])
df_acc.to_csv(par.base_file + '/acc_f0.csv')

'''
    train_ll, train_acc, train_num, test_ll, test_acc, test_num=PL_activation.main()
    train_ll_l.append(train_ll)
    train_acc_l.append(train_acc)
    train_num_l.append(train_num)
    test_ll_l.append(test_ll)
    test_acc_l.append(test_acc)
    test_num_l.append(test_num)
'''



#stop = timeit.default_timer()
#print('Time: ', stop - start)