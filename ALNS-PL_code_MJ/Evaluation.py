from ALNS_class import ALNS
from Network import reward_net_1,reward_net_2, reward_net_3, reward_net_4, reward_net_5
from Data import MyDataset
from Preference_learning import preference
from Sample import sample
from Record import p_record,eva_record,save_eva
import numpy as np
import pandas as pd
import torch
from Demand import gen_demand
import os
import par
if __name__ == "__main__":
    print('XXXXXXXXXXX evaluation XXXXXXXXXXXX')
    base_file=par.base_file
    if not os.path.exists(base_file + '\modal_share'):
        os.makedirs(base_file + '\modal_share')

    #a=0
    #while not os.path.exists(base_file +'\online_network1.pkl'):
        #a=0

    #r=torch.load(base_file +'\online_network.pkl')
    r = reward_net_5()
    r.load_state_dict(torch.load(base_file +'\offline_network.pkl'))
    r.eval()

    file_name = base_file + '\offline\Result'
    eva_file = base_file + '\evaluation\Result'
    evaluation_num = par.evaluation_num  # 10
    eva = np.zeros((3, evaluation_num, 8))
    #shipper_group = np.zeros(ALNS_request_number)
    for i in range(0,evaluation_num):
        request = np.load(file_name + str(i) + "/request_generate.npy")
        shipper_group = np.zeros(len(request[:,0]))
        #planning = ALNS(request, par.ALNS_iteration, multi_obj=1, bi_obj_cost_time=0, bi_obj_cost_preference=1,
                        #exp_number=i,
                        #file=eva_file, dnn=r, shipper_h=shipper_group)
        planning = ALNS(request, par.ALNS_iteration, multi_obj=1, bi_obj_cost_time=0, bi_obj_cost_preference=1,
                        exp_number=i,
                        file=eva_file, dnn=r, shipper_h=shipper_group)
        planning.real_main()

        eva = eva_record(i, file_name, eva_file, eva)  # read result+find pareto+record

    save_eva(eva, base_file)  # write in csv record
