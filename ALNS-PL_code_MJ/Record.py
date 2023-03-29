import numpy as np
import pandas as pd
from True_preference import shipper_reward
import par

import os
base_file=par.base_file

def p_record (log_likelihood,accuracy,mode,test_num, ll_l,acc_l,state_l,test_num_l):
    #mode: -1:initial, 0:offline, 1:online
    ll_l.append(log_likelihood)
    acc_l.append(accuracy)
    state_l.append(mode)
    test_num_l.append(test_num)
    return ll_l,acc_l,state_l,test_num_l

def eva_record(i,file_name,eva_file, eva):
     #calculate the indicators for pareto solutions
     cost0,time0,ep0,rp0= offline_solution(file_name,i)#cost, time for offline
     cost1, time1, ep1, rp1 = online_solution(eva_file, i, att1=1, att2=15)  # cost, p for offline
     eva[0, i, :]= np.array([np.max(cost0),np.max(cost1),
                             np.max(time0),np.max(time1),
                             np.max(ep0), np.max(ep1),
                             np.max(rp0),np.max(rp1)])

     eva[1, i, :]= np.array([np.min(cost0),np.min(cost1),
                             np.min(time0),np.min(time1),
                             np.min(ep0),np.min(ep1),
                             np.min(rp0),np.min(rp1)])

     eva[2, i, :]= np.array([np.mean(cost0),np.mean(cost1),
                             np.mean(time0),np.mean(time1),
                             np.mean(ep0),np.mean(ep1),
                             np.mean(rp0),np.mean(rp1)])
     return eva

def offline_solution(file_name,i):
    att1=1
    # calcultae the real preference
    preference_att=np.load(file_name + str(i) + "/All results/attribute.npy")
    real_preference=np.zeros(len(preference_att[:,0,0]))
    for j in range(len(preference_att[:,0,0])):
        shipper_h = np.zeros(len(preference_att[0, :, 0]))  # IMPORT FROM THE OFFLINE FILE
        preference_list = shipper_reward(preference_att[j, :, :], shipper_h)
        real_preference[j] = np.sum(preference_list)
    path = file_name + str(i) \
           + '/comparerequest_number100percentage0.72/obj_recordcomparerequest_number100percentage0.72' + str(i) + '.xlsx'
    obj_data = pd.read_excel(path, 'obj_record')
    obj_columns=obj_data.columns
    obj_data = obj_data.values[:, 1:]
    solution_num=len(obj_data[:, 0])
    index = np.arange(solution_num, dtype=int).reshape(-1, 1)
    data = np.concatenate((obj_data, index), axis=1)
    data = data[data[:, att1].argsort()]  # descend
    solution_index = []
    for j in range(0, 5):#select the best five solution
        solution_index.append(int(data[j, -1]))

    cost = obj_data[solution_index, 1]  # cost
    time = obj_data[solution_index, 2]  # time
    e_p = obj_data[solution_index, 15]  # etimated_preference
    r_p = real_preference[solution_index]  # real_preference
    # save solution
    offline_solution = obj_data[solution_index, :]
    offline = pd.DataFrame(offline_solution,columns=obj_columns[1:])
    offline['real_preference']=r_p
    offline.to_csv(base_file + '\offline_solutions'+str(i)+'.csv')

    # save mode_share

    mode_share = np.load(file_name + str(i) + "/All results/modal_share.npy")
    for j in range(len(solution_index)):
        mode = pd.DataFrame(mode_share[solution_index[j], :, :],
                            columns=['barge', 'train', 'truck'])
        mode.to_csv(base_file + '\modal_share\mode_offline' + str(i)+'s'+ str(j) + '.csv')

    return cost, time,e_p,r_p

def online_solution(file_name,i,att1,att2):# 1 cost,2 time, 15 calculated-preference,16 real-preference
    # file_name = base_file + '\offline\Result'
    # find the pareto solution among all the solutions
    path = file_name + str(i) \
           + '/comparerequest_number100percentage0.72/obj_recordcomparerequest_number100percentage0.72' + str(i) + '.xlsx'
    obj_data = pd.read_excel(path, 'obj_record')
    obj_columns = obj_data.columns
    obj_data = obj_data.values[:, 1:]
    solution_num=len(obj_data[:, 0])

    index = np.arange(solution_num, dtype=int).reshape(-1, 1)
    data = np.concatenate((obj_data, index), axis=1)
    data = data[data[:, att1].argsort()]  # ascend
    pareto_index = []
    pareto_index.append(int(data[0, -1]))
    y = data[0, att2]
    for j in range(1, solution_num):
        if data[j, att2] < y:
            pareto_index.append(int(data[j, -1]))
            y = data[j, att2]

    cost=obj_data[pareto_index,1] # cost
    time=obj_data[pareto_index,2] #time
    e_p= obj_data[pareto_index,15] #etimated_preference
    r_p = obj_data[pareto_index, 16]#real_preference
    # save solution
    online_solution = obj_data[pareto_index, :]
    online=pd.DataFrame(online_solution,columns=obj_columns[1:])
    online.to_csv(base_file + '\online_solutions'+str(i)+'.csv')

    mode_share=np.load(file_name + str(i) + "/All results/modal_share.npy")
    for j in range(len(pareto_index)):
        mode = pd.DataFrame(mode_share[pareto_index[j],:,:],
                              columns=['barge', 'train', 'truck'])
        mode.to_csv(base_file + '\modal_share\mode_online'+str(i)+'p'+str(j)+'.csv')

    return cost, time, e_p, r_p

def save_estimate(ll_l, acc_l, state_l, test_num_l,base_file):
    train=pd.DataFrame(columns=['LL','accuracy','state','test_num'])
    train['LL']=ll_l
    train['accuracy'] = acc_l
    train['state'] = state_l
    train['test_num']=test_num_l
    train.to_csv(base_file+'/training_record.csv')

def save_eva(eva,base_file):
    max=pd.DataFrame(eva[0,:,:],columns=['base_c','c',
                                         'base_t','t',
                                         'base_ep','ep',
                                         'base_rp','rp'])

    min=pd.DataFrame(eva[1,:,:],columns=['base_c','c',
                                         'base_t','t',
                                         'base_ep','ep',
                                         'base_rp','rp'])
    median=pd.DataFrame(eva[2,:,:],columns=['base_c','c',
                                         'base_t','t',
                                         'base_ep','ep',
                                         'base_rp','rp'])

    max.to_csv(base_file+'/pareto_max.csv')
    min.to_csv(base_file + '/pareto_min.csv')
    median.to_csv(base_file + '/pareto_median.csv')


def save_data(route_0, route_1,real_choice,shipper_group):
    n = len(route_1)
    df = pd.DataFrame()
    df['CHOICE'] = real_choice
    df['group'] = shipper_group

    df['cost_1'] = route_0[:, 0]
    df['time_1'] = route_0[:, 1]
    df['delay_1'] = route_0[:, 2]
    df['emission_1'] = route_0[:, 3]
    df['trans_1'] = route_0[:, 4]
    df['A1'] = np.ones(n)

    df['cost_2'] = route_1[:, 0]
    df['time_2'] = route_1[:, 1]
    df['delay_2'] = route_1[:, 2]
    df['emission_2'] = route_1[:, 3]
    df['trans_2'] = route_1[:, 4]
    df['A2'] = np.ones(n)
    return df