import numpy as np
import torch
import pandas as pd
from True_preference import shipper_reward,shipper_compare
import par
class sample( ):
    def __init__(self, interactive, r,  epsilon):
        self.interactive=interactive # 0 offline, 1 online
        self.r = r
        self.att1 = 1 #the index of cost
        self.epsilon=epsilon
        self.attribute_num =5

    def read(self,iteration,file_obj):
        '''
        path = 'result_sample_12.13/Result' + str(
            iteration) + '/percentage0.72parallel_number9dynamic0/obj_recordpercentage0.72parallel_number9dynamic0' + str(
            iteration) + '.xlsx'
        '''
        path =file_obj + str(
            iteration) + '/comparerequest_number100percentage0.72/obj_recordcomparerequest_number100percentage0.72'+ str(
            iteration) + '.xlsx'
        self.obj_data = pd.read_excel(path, 'obj_record')
        self.obj_data = self.obj_data.values[:, 1:]
        #self.attribute_matrix = np.load(file="result_sample_12.13/data_r50_i100" + str(iteration) + ".npy")
        #self.attribute_matrix = np.load(file= file_att + str(iteration) + ".npy")
        self.attribute_matrix = np.load(file = file_obj + str(iteration) + "/All results/attribute.npy")#/All results
        self.shipper_num= len(self.attribute_matrix[0, :, 0])
        self.solution_num = len(self.obj_data[:,0])
        #self.shipper_h_list = np.random.randint(par.category, size=self.shipper_num)
        self.shipper_h_list =np.random.choice(par.category, size=self.shipper_num, p=par.percentage)
        self.shipper_h_list=self.shipper_h_list .reshape((self.shipper_num,1))
        self.shipper_h = np.dstack([self.shipper_h_list] * self.solution_num)
        self.shipper_h = np.transpose(self.shipper_h, (2, 0, 1))
        self.attribute_matrix = np.concatenate((self.attribute_matrix, self.shipper_h), axis=-1)
        if self.interactive == 1:
            overall_preference = np.zeros(self.solution_num)
            for i in range(self.solution_num):
                overall_preference[i] = np.sum(self.r.estimated_reward(torch.tensor(self.attribute_matrix[i, :, :]).float()).detach().numpy())
            self.obj_data = np.concatenate((self.obj_data, overall_preference.reshape(-1, 1)), axis=1)
        return self.obj_data, self.attribute_matrix

    def find_solution (self): # find the alternative in offline: single objective
        data= self.obj_data
        index = np.arange(self.solution_num, dtype=int).reshape(-1, 1)
        data = np.concatenate((data, index), axis=1)
        data = data[data[:, self.att1].argsort()]  # descend
        solution_index = []
        for i in range(0, 50):  # select the best five solution
            solution_index.append(int(data[i, -1]))
        #for i in range(int(len(data[:, 0]) * self.epsilon)):  # do not repeat?
            #pareto_index.append(np.random.randint(self.solution_num))
        pareto_attribute = self.attribute_matrix[solution_index, :, :]
        return pareto_attribute

    def find_pareto(self):  # find the alternative in offline: single objective
        self.att2 = 15  #estimated_preference
        data=self.obj_data
        index = np.arange(self.solution_num, dtype=int).reshape(-1, 1)
        data = np.concatenate((data, index), axis=1)
        data = data[data[:, self.att1].argsort()]  # ascend
        pareto_index = []
        pareto_index.append(int(data[0, -1]))
        y = data[0, self.att2]
        for i in range(1, self.solution_num):# lower att2 means higher satisfaction
            if data[i, self.att2] < y:
                pareto_index.append(int(data[i, -1]))
                y = data[i, self.att2]
        pareto_attribute = self.attribute_matrix[pareto_index, :, :]
        return pareto_attribute


    def solution_index(self, shippers, solutions):  # traning=1 for training data, 0 for test data
        #pareto_solution_num=len(self.pareto_attribute[:,0,0])
        base = np.arange(solutions)
        solution_0 = np.repeat(base, solutions)
        solution_1 = np.tile(base, solutions)
        valid_index = solution_0 > solution_1
        solution_0 = solution_0[valid_index]
        solution_1 = solution_1[valid_index]
        solution_0 = np.tile(solution_0, shippers)
        solution_1 = np.tile(solution_1, shippers)
        return solution_0, solution_1


    def sampling(self,data):#[solution, shipper, attribute]
        solution_n = len(data[:,0,0])
        shipper_n= len(data[0,:,0])
        shipper_base = np.arange(int(shipper_n))  # len=number of shippers for training
        solution0_index, solution1_index = self.solution_index(shipper_n,solution_n)
        shipper_index = np.repeat(shipper_base, np.sum(np.arange(solution_n)))

        route_0 = np.reshape(data[solution0_index, shipper_index, :], (-1, self.attribute_num+1))
        route_1 = np.reshape(data[solution1_index, shipper_index, :], (-1, self.attribute_num+1))
        index = []
        # delete repeated rows
        route=np.zeros((len(route_0[:,0]),len(route_0[0,:])*2))
        route[:,:len(route_0[0,:])]=route_0
        route[:,len(route_0[0,:]):]=route_1
        route = np.unique(route, axis=0)
        #route_new = route[np.sum(route[:, :len(route_0[0,:])]) == np.sum(route[:, len(route_0[0,:]):]), :]

        route_0 = route[:,:len(route_0[0,:])]
        route_1 =  route[:,len(route_0[0,:]):]

        # delete repeated rows
        for i in range(len(route_0[:, 0])):
            if np.sum(route_0[i, :]) == np.sum(route_1[i, :]):
                index.append(i)

        route_0 = np.delete(route_0, index, axis=0)
        route_1 = np.delete(route_1, index, axis=0)

        route_h = route_0[:, -1]
        route_0 = route_0 [:,0:-1]
        route_1 = route_1[:, 0:-1]
        return route_0, route_1, route_h

    def main_sample(self,start_train_n,end_train_n, file_obj):
        '''
        :param end_train_n: the last index of planning result for training
        :param file_obj: input of read()
        :return: route_0,roue_1
        '''
        for i in range (start_train_n,end_train_n):
            self.read(i,file_obj)

            if self.interactive==1:
                train_attribute = self.find_pareto()
            else:
                train_attribute = self.find_solution()

            if i == start_train_n:
                route_0, route_1, route_h = self.sampling(train_attribute)
            else:
                new_route_0, new_route_1, new_route_h = self.sampling(train_attribute)
                route_0 = np.concatenate((route_0,new_route_0),axis=0)
                route_1 = np.concatenate((route_1, new_route_1), axis=0)
                route_h = np.concatenate((route_h, new_route_h), axis=0)

        train_num = len(route_0[:,0])
        return train_num, route_0, route_1, route_h

    def test_sample(self,file_obj, start_n,end_n):
        for i in range (start_n,end_n):
            self.read(i,file_obj)

            if self.interactive == 1:
                pareto_attribute = self.find_pareto()
            else:
                pareto_attribute = self.find_solution()
            # pareto_solution_num=len(pareto_attribute [:,0,0])
            test = pareto_attribute
            if i == start_n:
                test_route_0, test_route_1,test_route_h = self.sampling(test)
            else:
                new_test_route_0, new_test_route_1,new_test_route_h = self.sampling(test)
                test_route_0 = np.concatenate((test_route_0, new_test_route_0), axis=0)
                test_route_1 = np.concatenate((test_route_1, new_test_route_1), axis=0)
                test_route_h = np.concatenate((test_route_h, new_test_route_h), axis=0)

        return test_route_0, test_route_1,test_route_h

    def real_choice (self,route_0,route_1,shipper_h):
        test_reward_0 = shipper_reward(route_0,shipper_h)
        test_reward_1 = shipper_reward(route_1,shipper_h)
        real_compare = shipper_compare(test_reward_0, test_reward_1)
        return real_compare
