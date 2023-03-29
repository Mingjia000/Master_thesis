from ALNS_class import ALNS
import numpy as np
from Demand import gen_demand
from Network import reward_net_1,reward_net_2, reward_net_3, reward_net_4, reward_net_5

import os
if __name__ == "__main__":
    '''-----------offline ALNS-----------'''
    '''parameter setting'''
    ALNS_request_number=20
    ALNS_iteration = 20
    operation_num = 1 # instances number
    r = reward_net_5()
    shipper_group = np.zeros(ALNS_request_number)  # homogeneous
    # Mingjia_path
    base_file ='check_att'
    file_name = base_file + '\offline\Result'
    if not os.path.exists(base_file + '\offline'):
        os.makedirs(base_file + '\offline')

    '''ALNS'''
    for i in range (operation_num):
        request = gen_demand(ALNS_request_number)
        planning = ALNS(request,ALNS_iteration,multi_obj=0,bi_obj_cost_time=0,bi_obj_cost_preference=0,exp_number=i,file=file_name, dnn=r,shipper_h=shipper_group)
        planning.real_main()
        np.save(file=file_name + str(i) + "/request_generate.npy", arr=request)
        modal_share=np.load(file=file_name + str(i) + "/modal_share.npy")
