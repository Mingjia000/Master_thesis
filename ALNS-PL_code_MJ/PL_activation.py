from ALNS_class import ALNS
from Network import reward_net_1,reward_net_2, reward_net_3, reward_net_4, reward_net_5
from Preference_learning import preference
from Sample import sample
from Test_network import prediction_validation
from Benchmark import BL
from Record import p_record,eva_record,save_estimate,save_eva,save_data
import numpy as np
import torch
import pickle
from Demand import gen_demand
import os
import par
import shap
import pandas as pd

if __name__ == "__main__":
#def main(model):
    model=par.train_model
    '''-----------record-----------'''
    ll_l=[]
    acc_l=[]
    state_l=[]
    test_num_l=[]
    train_ll_l = []
    train_acc_l = []
    train_num_l = []
    test_ll_l = []
    test_acc_l = []
    test_num_l = []
    '''-----------offline ALNS-----------'''
    # Mingjia_path
    base_file =par.base_file
    file_name = base_file + '\offline\Result'

    '''-----------run the rest when passing while loop-----------'''

    if par.function ==0:
        function_path='/linear'
    elif par.function ==1:
        function_path = '/piecewise'
    else:
        function_path = '/nonlinear'

    if par.train_model == 0:
        model_path='_BL'
    elif par.train_model == 1:
        model_path = '_1NN'
    else:
        model_path = '_5NN'

    if not os.path.exists(base_file + '\learning'):
        os.makedirs(base_file + '\learning')
    if not os.path.exists(base_file + '\learning' + function_path):
        os.makedirs(base_file + '\learning' + function_path)
        #if not os.path.exists(base_file + '\evaluation'):
            #os.makedirs(base_file + '\evaluation')

    '''-----------offline learning-----------'''
    epsilon = 0
    interactive = 0  # interactive = 0, compare 5 top solutions; interactive = 1, compare pareto solutions
    end_train_n = par.end_train_n
    start_test_n = par.start_test_n #[start_test_n,end_test_n)
    end_test_n = par.end_test_n
    #model= model # 0:BL; 1:one-layer, 2: 5-layer
    #output all the data, if use 10-fold learning how to use stages

    '''--test sample --'''
    r=0
    s = sample(interactive, r, epsilon)  # initialize offline sampling
    test_route_0, test_route_1, test_route_h = s.test_sample(file_name, start_test_n, end_test_n)
    test_real_compare = s.real_choice(test_route_0, test_route_1,test_route_h)
    df_test=save_data(test_route_0, test_route_1, test_real_compare, test_route_h)
    df_test.to_csv(base_file + '\learning' + function_path+'/test_data.csv')
    '''--train sample--'''
    route_0, route_1,route_h = s.test_sample(file_name, 0, end_train_n)
    real_compare = s.real_choice(route_0,route_1,route_h)
    df=save_data(route_0, route_1, real_compare, route_h)
    df.to_csv(base_file + '\learning' + function_path + '/train_data.csv')

    if model ==0:
        path = base_file + '\learning' + function_path
        #bio, save bie
        '''--train --'''
        b = BL(df)
        pickleFile = base_file + '/BL.pickle'
        dcm_model,betas = b.train()
        #for k, v in betas.items():
            #print(f"{k:10}=\t{v:.3g}")
        pickle.dump(dcm_model, open(pickleFile, 'wb'))
        '''--initialize prediction and validation --'''
        pv = prediction_validation(test_real_compare,model)
        accuracy = pv.validate_dcm(df_test,pickleFile)
        print(accuracy)
        #return accuracy, betas

    elif model ==1:
        r = reward_net_1()
    else:
        r = reward_net_5()

    if model != 0:
        '''parameter settings for preference learning'''
        offline_i = par.offline_i # can be divided by the result number for training #18
        learning_rate = 0.001 #linear 0.01
        epochs = 30
        batch_size = 32*16
        # 10 weeks, each week ,shipper个数不同，训练集和不同
        #一周一周训练 直到达标
        #记录一下每一个week(内部acc)的情况
        #if 达到70%， 变动小于1% ， 开始取一个最高的
        #用最高的开始online leanring
        pv = prediction_validation(test_real_compare,model)

        for i in range (int((end_train_n+1)/offline_i)):
            start = offline_i * i
            end =offline_i* (i+1)
            train_num, route_0, route_1,route_h = s.main_sample(start,end, file_name)  # sampling
            print(train_num,start,end,np.unique(route_h))
            '''----output training data----'''
            pl = preference(train_num, learning_rate, epochs, batch_size, r) # initial a instance for preference learning
            #ll_l, acc_l, state_l, test_num_l= pl.train(route_0, route_1,shipper_group) # DNN training
            pl.train(route_0, route_1,route_h) # DNN training
            #train_ll_l, train_acc_l, train_num_l, test_ll_l, test_acc_l, test_num_l= pl.train(route_0, route_1,shipper_group)
            #t = test(r) # testing
            #log_likelihood, accuracy=t.test(test_route_0, test_route_1,test_shipper_group)
            #ll_l, acc_l, state_l, test_num_l = p_record(log_likelihood, accuracy, 0,test_num, ll_l, acc_l, state_l,test_num_l)  # -1:initial state

        accuracy = pv.validate_nn(r,test_route_0,test_route_1)
        print(accuracy)
        #return accuracy, 0
        torch.save(r.state_dict(), base_file + '\learning' + function_path + '\offline_network.pkl')
            #return train_ll_l, train_acc_l, train_num_l, test_ll_l, test_acc_l, test_num_l
            #test_state_0 = torch.from_numpy(test_route_0).float()
            #e = shap.DeepExplainer(r, test_state_0)
            #shap_values = e.shap_values(test_state_0)
            #shap.initjs()
            #shap.summary_plot(shap_values,test_route_0)


    '''
    
    #-----------online learning-----------
    print('XXXXXXXXXXX online learning XXXXXXXXXXXX')
    J = par.J # iteration number for ALNS-PL-ALNS- #10
    interactive = 1
    file_name = base_file + '\online\Result'
    ALNS_request_number = np.random.randint(low=par.r_min, high=par.r_max, size=J)*10 # 50-100#

    for j in range (0,J):
        print('ALNS-PL-online_'+str(j))
        #---ALNS---
        request = gen_demand(ALNS_request_number[j])
        exp_number = j
        shipper_group = np.zeros(ALNS_request_number[j])  # homogeneous
        planning = ALNS(request, par.ALNS_iteration, multi_obj=1, bi_obj_cost_time=0, bi_obj_cost_preference=1,
                        exp_number=j, file=file_name, dnn=r,shipper_h=shipper_group)
        planning.real_main()
        np.save(file=file_name + str(j) + "/request_generate.npy", arr=request)

        start = j
        end = j+1
        train_num, route_0, route_1 = s.main_sample(start,end, file_name)  # sampling
        shipper_group=np.zeros(len(route_0[:,0])) #homogeneous
        pl = preference(train_num, learning_rate, epochs, batch_size, r) # initial a instance for preference learning
        pl.train(route_0, route_1,shipper_group) # DNN training
        # testing
        t = test(r)
        log_likelihood, accuracy=t.test(test_route_0, test_route_1,test_shipper_group)
        ll_l, acc_l, state_l, test_num_l = p_record(log_likelihood, accuracy, 1,test_num, ll_l, acc_l, state_l,test_num_l)  # -1:initial state

    torch.save(r.state_dict(), base_file+'\online_network.pkl')
    
    '''
    #save_estimate(ll_l, acc_l, state_l, test_num_l,base_file)
