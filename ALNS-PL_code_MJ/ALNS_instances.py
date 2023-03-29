from ALNS_class import ALNS
import numpy as np
from Demand import gen_demand
from Network import reward_net_1,reward_net_2, reward_net_3, reward_net_4, reward_net_5
import par
import os
if __name__ == "__main__":
    '''-----------offline ALNS-----------'''
    '''parameter setting'''
    ALNS_iteration = par.ALNS_iteration
    operation_num = par.operation_num # instances number 30
    r = reward_net_5()
    # Mingjia_path
    base_file =par.base_file
    file_name = base_file + '\offline\Result'

    if not os.path.exists(base_file + '\offline'):
        os.makedirs(base_file + '\offline')

    '''ALNS'''
    for i in range (0,operation_num):
        ALNS_request_number = np.random.randint(low=par.r_min, high=par.r_max, size=1) * 10  # 50-100
        shipper_group = np.zeros(ALNS_request_number)  # homogeneous
        request = gen_demand(ALNS_request_number)
        planning = ALNS(request,ALNS_iteration,multi_obj=0,bi_obj_cost_time=0,bi_obj_cost_preference=0,exp_number=i,file=file_name, dnn=r,shipper_h=shipper_group)
        planning.real_main()
        np.save(file=planning.user+'/'+file_name + str(i) + "/request_generate.npy", arr=request)


