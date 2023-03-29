
base_file='ALNS_PL_3.28'#'midterm'#'ALNS_PL_MJ_3.03_4'
r_min=10 #r_min*10
r_max=20

#offline
operation_num=50
ALNS_iteration=200
function=0 #0:linear,1:piecewise, 2:nonlinear
train_model=2 #0=BL,2=5layerNN
category=1# class of shippers 1=homo,2,3,4
if category == 1:
    percentage=[1]
else:
    percentage=[0.25,0.25,0.25,0.25]
#online
J=0
end_train_n = 25 # 0-end_train_n #17
start_test_n = 25 # [start_test_n,end_test_n)
end_test_n = 30  # 21

#offline_i = end_train_n+1  # can be divided by the result number for training
offline_i = 25

#evaluation
evaluation_num=1