import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from Data import  MyDataset
from True_preference import shipper_reward,shipper_compare
from Test_network import prediction_validation
from Record import p_record
from Sample import sample
import par
import pandas as pd
class preference( ):
    def __init__(self,num, learning_rate,epochs,batch_size,  r):
        '''
        :param num:
        :param learning_rate: learning_rate of dnn
        :param epochs:
        :param batch_size:
        :param data: route [solution, shippers, attribute]
        :param train_p: the proportion of routes used as training data
        '''

        self.num=num
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        #for i in range (len(data[:,0,0])):
            #self.data[i, :, 0:2] = preprocessing.normalize(data[i, :, 0:2], norm='l2', axis=1)
        self.R = r
        #self.Dataset = MyDataset()

        self.optimizer = torch.optim.AdamW(self.R.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_MSE = nn.MSELoss()

        #self.shipper_num = len(data[0,:,0])
        #self.solution_num = len(data[:,0,0])
        #self.attribute_num = len(data[0,0, :])
    '''

    def solution_index (self,shippers):#traning=1 for training data, 0 for test data
        base=np.arange(self.solution_num)
        solution_0 = np.repeat(base, self.solution_num)
        solution_1=np.tile(base,self.solution_num)
        valid_index= solution_0 > solution_1
        solution_0 = solution_0 [valid_index]
        solution_1 = solution_1[valid_index]
        solution_0 =np.tile(solution_0 , shippers)
        solution_1 = np.tile(solution_1, shippers)
        return solution_0, solution_1

    def sampling (self):

        shipper_base = np.arange(int(self.shipper_num*self.train_p)) #len=number of shippers for training
        solution0_index, solution1_index = self.solution_index(len(shipper_base))
        shipper_index =np.repeat(shipper_base, np.sum(np.arange(self.solution_num)))

        route_0 =  np.reshape(self.data[solution0_index, shipper_index,:], (-1,self.attribute_num))
        route_1 = np.reshape(self.data[solution1_index,shipper_index,:],  (-1,self.attribute_num))
        index=[]
        #delete repeated rows
        for i in range (len(route_0[:,0])):
            if route_0[i,:].all() ==route_1[i,:].all():
                index.append(i)

        route_0=np.delete(route_0,index,axis=0)
        route_1 = np.delete(route_1, index, axis=0)

        self.num = len(route_0[:,0])
        print('number of training data',self.num)

        test_shipper_base = np.arange(start=int(self.shipper_num*self.train_p), stop=int(self.shipper_num)) #len=number of shippers for testing
        test_solution0_index, test_solution1_index = self.solution_index(len(test_shipper_base))
        test_shipper_index =np.repeat(test_shipper_base,np.sum(np.arange(self.solution_num)))


        test_route_0 =  np.reshape(self.data[test_solution0_index,test_shipper_index,:],  (-1,self.attribute_num))
        test_route_1 = np.reshape(self.data[test_solution1_index,test_shipper_index,:],  (-1,self.attribute_num))

        index=[]
        for i in range (len(test_route_0[:,0])):
            if test_route_0[i,:].all() ==test_route_1[i,:].all():
                index.append(i)

        test_route_0 = np.delete(test_route_0,index,axis=0)
        test_route_1 = np.delete(test_route_1, index, axis=0)
        self.test_num = len(test_route_0[:, 0])
        print('number of testing data',self.test_num)

        return route_0, route_1, test_route_0, test_route_1

'''
    def calculate_estimated_result(self,x_0,x_1):
        estimated_compare = torch.zeros(self.batch_size)
        estimated_r_0 = torch.zeros(self.batch_size)
        estimated_r_1 = torch.zeros(self.batch_size)

        for i in range(self.batch_size):
            state_0 = x_0[i, :]
            state_1 = x_1[i, :]
            estimated_r_0[i] = self.R(state_0)
            estimated_r_1[i] =self.R(state_1)
            if estimated_r_0[i] > estimated_r_1[i]:
                estimated_compare[i] = 0
            elif estimated_r_0[i] < estimated_r_1[i]:
                estimated_compare[i] = 1
            else:
                estimated_compare[i] = 0.5

        estimated_result = [estimated_r_0, estimated_r_1, estimated_compare]

        return estimated_result

    def train(self, route_0, route_1,shipper_h):
        # route_0,route_1, test_route_0, test_route_1  = self.sampling()
        reward_0 = shipper_reward(route_0,shipper_h)
        reward_1 = shipper_reward(route_1,shipper_h)
        real_compare = shipper_compare(reward_0, reward_1)

        train_dataset = MyDataset(route_0, route_1, real_compare)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, self.batch_size, shuffle=True, drop_last=True)
        loss_list = []
        ll_l = []
        acc_l = []
        state_l = []
        test_num_l = []
        train_ll_l = []
        train_acc_l = []
        train_state_l = []
        train_num_l = []
        for epoch in range(self.epochs):
            loss_sum = 0
            for step, (batch_x_0, batch_x_1, batch_y) in enumerate(train_dataloader):
                estimated_y = self.calculate_estimated_result(batch_x_0.float(), batch_x_1.float())
                estimated_y[0] = torch.reshape(estimated_y[0], (self.batch_size, -1))  # value for route_0
                estimated_y[1] = torch.reshape(estimated_y[1], (self.batch_size, -1))  # value for route_1

                input = torch.cat((estimated_y[0], estimated_y[1]), 1)
                target = batch_y.long()
                loss = self.criterion(input, target)
                #print(loss)

                # take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_sum = loss + loss_sum
                '''

                #acc on train set
                t = test(self.R)  # testing
                train_shipper_group = np.zeros(len(batch_x_0[:, 0]))
                train_log_likelihood, train_accuracy = t.test(batch_x_0.numpy(), batch_x_1.numpy(), train_shipper_group)
                train_ll_l, train_acc_l, train_state_l, train_num_l = p_record(train_log_likelihood, train_accuracy, 0, len(batch_x_0[:, 0]),
                                                            train_ll_l, train_acc_l, train_state_l,
                                                            train_num_l)  # -1:initial state
                # acc on test set
                ll_l, acc_l, state_l, test_num_l = self.test_batch(ll_l, acc_l, state_l, test_num_l)
                '''
            #print('train_',train_acc_l)
            #print('test_', acc_l)
            #print('nn', loss_sum.detach().numpy() / (self.num / self.batch_size))
            print('nn loss', loss_sum / (self.num / self.batch_size))
            #loss_sum = torch.tensor(loss_sum)
            #loss_list.append(loss_sum.detach().numpy() / (self.num / self.batch_size))
        #return  ll_l, acc_l, state_l, test_num_l
        #self.save_batch(train_ll_l, train_acc_l, train_state_l, train_num_l, ll_l, acc_l, state_l, test_num_l, par.base_file)
        #return max(train_ll_l), max(train_acc_l), max(train_num_l),max(ll_l), max(acc_l), max(test_num_l)

    def test_batch(self, ll_l, acc_l, state_l,test_num_l):

        r=self.R
        epsilon=0
        interactive = 0  # as input to sample, whether to use DNN
        start_test_n = par.start_test_n  # [start_test_n,end_test_n)
        end_test_n = par.end_test_n  # 21
        file_name = par.base_file + '\offline\Result'

        self.s = sample(interactive, r, epsilon)  # initialize offline sampling
        test_num, test_route_0, test_route_1 = self.s.test_sample(file_name, start_test_n, end_test_n)
        test_shipper_group = np.zeros(len(test_route_0[:, 0]))

        t = test(self.R)  # testing
        log_likelihood, accuracy = t.test(test_route_0, test_route_1, test_shipper_group)
        ll_l, acc_l, state_l, test_num_l = p_record(log_likelihood, accuracy, 0, test_num,
                                                    ll_l, acc_l, state_l,
                                                    test_num_l)  # -1:initial state

        return ll_l, acc_l, state_l, test_num_l

    def save_batch(self, train_ll_l, train_acc_l, train_state_l, train_num_l, ll_l, acc_l, state_l, test_num_l, base_file):
        acc = pd.DataFrame(columns=['LL', 'accuracy', 'state', 'test_num','train_LL', 'train_accuracy', 'train_state', 'train_num'])
        acc['test_LL'] = ll_l
        acc['train_LL'] = train_ll_l
        acc['test_accuracy'] = acc_l
        acc['train_accuracy'] = train_acc_l
        acc['test_state'] = state_l
        acc['train_state'] = train_state_l
        acc['test_test_num'] = test_num_l
        acc['train_test_num'] = train_num_l
        acc.to_csv(base_file + '/batch_record.csv')

'''
    def train (self,route_0,route_1, test_route_0, test_route_1):

        #route_0,route_1, test_route_0, test_route_1  = self.sampling()
        reward_0 = self.shipper_reward(route_0)
        reward_1 = self.shipper_reward(route_1)
        real_compare = self.shipper_compare(reward_0, reward_1)

        test_reward_0 = self.shipper_reward(test_route_0)
        test_reward_1 = self.shipper_reward(test_route_1)
        test_real_compare = self.shipper_compare(test_reward_0, test_reward_1)

        train_dataset = MyDataset(route_0, route_1, real_compare)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, self.batch_size, shuffle=True, drop_last=True)
        loss_list = []

        test_estimated_r_0 = torch.zeros(self.test_num)
        test_estimated_r_1 = torch.zeros(self.test_num)
        test_estimated_compare = np.array([0 for i in range(self.test_num)])
        incorrect = 0
        incorrect_list= []


        for i in range(len(test_route_0[:, 0])):
            test_state_0 = torch.from_numpy(test_route_0[i, :]).float()
            test_state_1 = torch.from_numpy(test_route_1[i, :]).float()

            test_estimated_r_0[i] = self.R.estimated_reward(test_state_0)
            test_estimated_r_1[i] = self.R.estimated_reward(test_state_1)

            if test_estimated_r_0[i] > test_estimated_r_1[i]:
                test_estimated_compare[i] = 0
                if test_real_compare[i] != 0:
                    incorrect = incorrect + 1
            elif test_estimated_r_0[i] < test_estimated_r_1[i]:
                test_estimated_compare[i] = 1
                if test_real_compare[i] != 1:
                    incorrect = incorrect + 1
            else:
                test_estimated_compare[i] = 0.5
                if test_real_compare[i] != 0.5:
                    incorrect = incorrect + 1
        print('accuracy before trainig',incorrect / self.test_num)

        for epoch in range(self.epochs):
            loss_sum = 0
            for step, (batch_x_0, batch_x_1, batch_y) in enumerate(train_dataloader):
                estimated_y = self.calculate_estimated_result(batch_x_0.float(), batch_x_1.float())
                estimated_y[0] = torch.reshape(estimated_y[0], (self.batch_size, -1)) #value for route_0
                estimated_y[1] = torch.reshape(estimated_y[1], (self.batch_size, -1)) #value for route_1

                input = torch.cat((estimated_y[0], estimated_y[1]), 1)
                target = batch_y.long()
                loss = self.criterion(input, target)

                # take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_sum = loss + loss_sum

            #print('nn', loss_sum.detach().numpy() / (self.num / self.batch_size))
            print('nn loss', loss_sum / (self.num / self.batch_size))
            loss_sum=torch.tensor(loss_sum)
            loss_list.append(loss_sum.detach().numpy() / (self.num / self.batch_size))

        incorrect=0

        for i in range(len(test_route_0[:, 0])):
            test_state_0 = torch.from_numpy(test_route_0[i, :]).float()
            test_state_1 = torch.from_numpy(test_route_1[i, :]).float()

            test_estimated_r_0[i] = self.R.estimated_reward(test_state_0)
            test_estimated_r_1[i] = self.R.estimated_reward(test_state_1)

            if test_estimated_r_0[i] > test_estimated_r_1[i]:
                test_estimated_compare[i] = 0
                if test_real_compare[i] != 0:
                    incorrect = incorrect + 1
            elif test_estimated_r_0[i] < test_estimated_r_1[i]:
                test_estimated_compare[i] = 1
                if test_real_compare[i] != 1:
                    incorrect = incorrect + 1
            else:
                test_estimated_compare[i] = 0.5
                if test_real_compare[i] != 0.5:
                    incorrect = incorrect + 1
        print('accuracy after trainig',incorrect / self.test_num)

        return loss_list

'''


