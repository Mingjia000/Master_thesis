import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.results as res
from biogeme.expressions import Beta, DefineVariable
import shap
import biogeme.expressions as ex
import matplotlib.pyplot as plt
import seaborn as sns


def decision_maker(r1, r2, real_compare, shipper_num, shipper):
    dm = np.zeros((shipper_num, 5))
    for i in range(len(real_compare)):
        shipper_id = shipper[i]
        if real_compare[i] == 0:
            for j in range(5):
                if r1[i, j] < r2[i, j]:
                    dm[shipper_id, j] = dm[shipper_id, j] + 1
                else:
                    dm[shipper_id, j] = dm[shipper_id, j] - 1
        elif real_compare[i] == 1:
            for j in range(5):
                if r1[i, j] > r2[i, j]:
                    dm[shipper_id, j] = dm[shipper_id, j] + 1
                else:
                    dm[shipper_id, j] = dm[shipper_id, j] - 1

    sum = np.sum(np.abs(dm), axis=1)
    dm = dm / sum[:, np.newaxis]
    return dm


class MyDataset(Dataset):
    def __init__(self, x_0, x_1, y):
        self.x_0_data = x_0
        self.x_1_data = x_1
        self.y_data = y
        self.length = len(self.y_data)

    def __getitem__(self, index):
        return self.x_0_data[index, :, :], self.x_1_data[index, :, :], self.y_data[index]

    def __len__(self):
        return self.length


# estimate reward, comparison by network
def calculate_estimated_result(x_0, x_1, batch_size, r):
    estimated_compare = torch.zeros(batch_size)
    estimated_r_0 = torch.zeros(batch_size)
    estimated_r_1 = torch.zeros(batch_size)

    state_0 = x_0
    state_1 = x_1
    estimated_r_0 = r(state_0)
    estimated_r_1 = r(state_1)
    for i in range(batch_size):
        if estimated_r_0[i] > estimated_r_1[i]:
            estimated_compare[i] = 0
        elif estimated_r_0[i] < estimated_r_1[i]:
            estimated_compare[i] = 1
        else:
            estimated_compare[i] = 0.5

    estimated_result = [estimated_r_0, estimated_r_1, estimated_compare]

    return estimated_result


# simulate shippers' comparison on transport plans
def shipper_compare(reward_0, reward_1):
    compare = np.array([0 for i in range(len(reward_1))])
    for i in range(len(reward_1)):
        if reward_0[i] > reward_1[i]:
            compare[i] = 0
        elif reward_0[i] < reward_1[i]:
            compare[i] = 1
        else:
            compare[i] = 0
    return compare


def shipper_reward(route, h):
    reward = np.zeros(len(route[:, 0]))
    mu, beta = 0, 1

    s = np.random.gumbel(mu, beta, len(route[:, 0]))
    '''

    c1 = np.percentile(route[:, 0], 30)
    c2 = np.percentile(route[:, 0], 60)
    theta2 = 3 / 2 * np.log(c1) ** 2
    theta3 = 3 * np.log(c1) * np.log(c2)
    gamma2 = -0.5 * np.log(c1) ** 3
    gamma3 = -0.5 * np.log(c1) * (3 * np.log(c2) ** 2 + np.log(c1) ** 2)
    reward = reward.astype(float)

    for i in range(len(route[:, 0])):
        if h[i] == 0:
            r0= -10
            r1= - 8 * 2.5* 5
            r2=- 5 * 5* 5* 5
            r3=- 2 * 5
            r4= - 2
        elif h[i] == 1:
            r0= -10
            r1= - 8
            r2=- 5
            r3=- 2 * 5* 5* 5* 5
            r4= - 2* 5
        elif h[i] == 2:
            r0= -10
            r1= - 8 * 5
            r2=- 5 * 5
            r3=- 2
            r4= - 2* 2.5* 5* 5* 5* 5
        else:
            r0= -10 * 5* 5
            r1= - 8 * 5* 5* 5
            r2=- 5
            r3=- 2
            r4= - 2

        if route[i, 0] < c1:
            cost = np.log(route[i, 0]) ** 3
        elif c1 < route[i, 0] < c2:
            cost = theta2 * np.log(route[i, 0]) ** 2 + gamma2
        else:
            cost = theta3 * np.log(route[i, 0]) + gamma3

        reward[i] = r0 * cost - r1 * route[i, 1] - r2 * route[i, 2] - r3 * route[i, 3] - r4 * \
                    route[i, 4]  # -0.0702*route[i,2]
    '''
    for i in range(len(route[:, 0])):
        if h[i] == 0:
            reward[i] = -10 * route[i, 0] - 8 * 5 * route[i, 1] - 5 * 5 * route[i, 2] - 2 * 5 * route[i, 3] - 2 * route[
                i, 4]
            #reward[i] = -10 * route[i, 0]
        elif h[i] == 1:
            reward[i] = -10 * 5 * route[i, 0] - 8 * route[i, 1] - 5 * route[i, 2] - 2 * route[i, 3] - 2 * route[i, 4]
        elif h[i] == 2:
            reward[i] = -10 * route[i, 0] - 8 * 2.5 * route[i, 1] - 5 * 2.5 * route[i, 2] - 2 * 5 * route[
                i, 3] - 2 * 5 * route[i, 4]
        elif h[i] == 3:
            reward[i] = -10 * 5 * route[i, 0] - 8 * 5 * route[i, 1] - 5 * 5 * route[i, 2] - 2 * route[i, 3] - 2 * 5 * \
                        route[i, 4]

    reward = reward + s
    return reward


def train_utility(train_dataloader, epochs, batch_size, r, learning_rate, s):
    optimizer = torch.optim.AdamW(r.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        loss_sum = 0
        for step, (batch_x_0, batch_x_1, batch_y) in enumerate(train_dataloader):
            x0 = batch_x_0[:, :, 0].float()
            x1 = batch_x_1[:, :, 0].float()
            u1 = r(x0)
            u2 = r(x1)
            latent = s(batch_x_0[:, :, 1].float())
            utility1 = u1[:, 0] * latent[:, 0] + u1[:, 1] * latent[:, 1] + u1[:, 2] * latent[:, 2] + u1[:, 3] * latent[
                                                                                                                :, 3]
            utility2 = u2[:, 0] * latent[:, 0] + u2[:, 1] * latent[:, 1] + u2[:, 2] * latent[:, 2] + u2[:, 3] * latent[
                                                                                                                :, 3]
            input = torch.cat((utility1.reshape(-1, 1), utility2.reshape(-1, 1)), 1)

            target = batch_y.long()
            loss = criterion(input, target)
            # take gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum = loss + loss_sum

        # print(epoch, 'nn',loss_sum/(train_n/batch_size)) #real
        if (epoch + 1) % 10 == 0:
            print("Epoch [{}/100], Loss: {:.4f}".format(epoch + 1, loss_sum / (train_n / batch_size)))
    return r


def train_seg(train_dataloader, epochs, batch_size, r, learning_rate, s):
    optimizer = torch.optim.AdamW(s.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        loss_sum = 0
        for step, (batch_x_0, batch_x_1, batch_y) in enumerate(train_dataloader):
            x0 = batch_x_0[:, :, 0].float()
            x1 = batch_x_1[:, :, 0].float()
            u1 = r(x0)
            u2 = r(x1)
            latent = s(batch_x_0[:, :, 1].float())
            utility1 = u1[:, 0] * latent[:, 0] + u1[:, 1] * latent[:, 1] + u1[:, 2] * latent[:, 2] + u1[:, 3] * latent[
                                                                                                                :, 3]
            utility2 = u2[:, 0] * latent[:, 0] + u2[:, 1] * latent[:, 1] + u2[:, 2] * latent[:, 2] + u2[:, 3] * latent[
                                                                                                                :, 3]
            input = torch.cat((utility1.reshape(-1, 1), utility2.reshape(-1, 1)), 1)
            target = batch_y.long()
            loss = criterion(input, target)
            # take gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum = loss + loss_sum

        # print(epoch, 'nn',loss_sum/(train_n/batch_size)) #real
        if (epoch + 1) % 10 == 0:
            print("Epoch [{}/100], Loss: {:.4f}".format(epoch + 1, loss_sum / (train_n / batch_size)))
    return r


def test(t1, t2, test_n, test_real_compare, r, s):
    incorrect = 0
    index = []
    c_index = []
    u_test_state_0 = torch.from_numpy(t1[:, :, 0]).float()
    u_test_state_1 = torch.from_numpy(t2[:, :, 0]).float()
    s_test_state = torch.from_numpy(t1[:, :, 1]).float()

    u1 = r(u_test_state_0)
    u2 = r(u_test_state_1)
    latent = s(s_test_state)
    utility1 = u1[:, 0] * latent[:, 0] + u1[:, 1] * latent[:, 1] + u1[:, 2] * latent[:, 2] + u1[:, 3] * latent[:, 3]
    utility2 = u2[:, 0] * latent[:, 0] + u2[:, 1] * latent[:, 1] + u2[:, 2] * latent[:, 2] + u2[:, 3] * latent[:, 3]

    for i in range(len(latent)):
        if utility1[i] > utility2[i] and test_real_compare[i] == 1:
            incorrect = incorrect + 1
            index.append(test_h[i])
        elif utility1[i] < utility2[i] and test_real_compare[i] == 0:
            incorrect = incorrect + 1
            index.append(test_h[i])
        else:
            c_index.append(test_h[i])

    print('accuracy', 1 - incorrect / test_n)
    # print('predicted right(class mean)',sum(index)/len(index))
    # print('predicted wrong(class mean)',sum(c_index)/len(c_index))


class reward_net(nn.Module):
    def __init__(self, state_dim=5, action_dim=5):
        super(reward_net, self).__init__()
        self.linear1 = nn.Linear(state_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, action_dim)
        self.m = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class seg_net(nn.Module):
    def __init__(self, state_dim=5, action_dim=5):
        super(seg_net, self).__init__()
        self.linear1 = nn.Linear(state_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, action_dim)
        self.m = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        # x = self.m(x)
        return x

class one_net(nn.Module):
    def __init__(self, state_dim=10, action_dim=1):
        super(one_net, self).__init__()
        self.linear1 = nn.Linear(state_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, action_dim)
        #self.m = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        # x = self.m(x)
        return x

class base(nn.Module):
    def __init__(self, state_dim=5, action_dim=1):
        super(base, self).__init__()
        self.linear1 = nn.Linear(state_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, action_dim)
        self.m = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

train_n = 4000
test_n = 200
shipper_num = 30
# 2 class
shipper = np.random.randint(shipper_num, size=train_n)
shipper_h = np.random.randint(1, size=shipper_num)
h = shipper_h[shipper]

test_shipper = np.random.randint(shipper_num, size=test_n)
test_h = shipper_h[test_shipper]

r1 = np.random.rand(train_n, 5)  # (train_n,4)
r2 = np.random.rand(train_n, 5)  # (train_n,4)

t1 = np.random.rand(test_n, 5)  # (test_n,4)
t2 = np.random.rand(test_n, 5)  # (test_n,4)

reward_0 = shipper_reward(r1, h)
reward_1 = shipper_reward(r2, h)
real_compare = shipper_compare(reward_0, reward_1)
dm = decision_maker(r1, r2, real_compare, shipper_num, shipper)
x1 = np.zeros((train_n, 5, 2))
x2 = np.zeros((train_n, 5, 2))
x1[:, :, 0] = r1
x1[:, :, 1] = dm[shipper, :]
x2[:, :, 0] = r2
x2[:, :, 1] = dm[shipper, :]

test_reward_0 = shipper_reward(t1, test_h)
test_reward_1 = shipper_reward(t2, test_h)
test_real_compare = shipper_compare(test_reward_0, test_reward_1)
test_dm = decision_maker(t1, t2, test_real_compare, shipper_num, test_shipper)
e = test_dm[test_shipper, :]

batch_size = 64 * 8
train_dataset = MyDataset(x1, x2, real_compare)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
learning_rate = 0.004  # 0.001
epochs = 400
round = 10
r = reward_net()
s = seg_net()

e1 = np.zeros((test_n, 5, 2))
e2 = np.zeros((test_n, 5, 2))
e1[:, :, 0] = t1
e1[:, :, 1] = dm[test_shipper, :]
e2[:, :, 0] = t2
e2[:, :, 1] = dm[test_shipper, :]
test(e1, e2, test_n, test_real_compare, r, s)
criterion = nn.CrossEntropyLoss()
def seg():
    for i in range(round):
        train_utility(train_dataloader, epochs, batch_size, r, learning_rate, s)
        test(e1, e2, test_n, test_real_compare, r, s)
        train_seg(train_dataloader, epochs, batch_size, r, learning_rate, s)
        test(e1, e2, test_n, test_real_compare, r, s)
        #train_utility(train_dataloader, epochs, batch_size, r, learning_rate, s)
        #test(e1, e2, test_n, test_real_compare, r, s)

def one():
    o=one_net()
    optimizer = torch.optim.AdamW(o.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        loss_sum = 0
        for step, (batch_x_0, batch_x_1, batch_y) in enumerate(train_dataloader):
            input0 = torch.cat((batch_x_0[:, :, 0].float(), batch_x_0[:, :, 1].float()), 1)
            input1 = torch.cat((batch_x_1[:, :, 0].float(), batch_x_1[:, :, 1].float()), 1)
            input0.requires_grad = True
            u1 = o(input0)
            u2 = o(input1)
            utility1 = u1
            utility2 = u2

            input = torch.cat((utility1.reshape(-1, 1), utility2.reshape(-1, 1)), 1)
            target = batch_y.long()
            loss = criterion(input, target)
            # take gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum = loss + loss_sum

        # print(epoch, 'nn',loss_sum/(train_n/batch_size)) #real
        if (epoch + 1) % 50 == 0:
            print("Epoch [{}/100], Loss: {:.4f}".format(epoch + 1, loss_sum / (train_n / batch_size)))

            incorrect = 0
            index = []
            c_index = []
            input0 = torch.cat((torch.from_numpy(e1[:, :, 0]).float(),torch.from_numpy(e1[:, :, 1]).float()), 1)
            input1 = torch.cat((torch.from_numpy(e2[:, :, 0]).float(), torch.from_numpy(e2[:, :, 1]).float()), 1)
            u1 = o(input0)
            u2 = o(input1)

            utility1 = u1
            utility2 = u2

            for i in range(len(e1[:, 0, 0])):
                if utility1[i] > utility2[i] and test_real_compare[i] == 1:
                    incorrect = incorrect + 1
                    index.append(test_h[i])
                elif utility1[i] < utility2[i] and test_real_compare[i] == 0:
                    incorrect = incorrect + 1
                    index.append(test_h[i])
                else:
                    c_index.append(test_h[i])

            print('accuracy', 1 - incorrect / test_n)

    input0 = torch.cat((torch.from_numpy(e1[:, :, 0]).float(), torch.from_numpy(e1[:, :, 1]).float()), 1)
    input0.requires_grad = True
    u1 = o(input0)
    grad_output = torch.ones_like(u1)
    grad_input = torch.autograd.grad(u1, input0, grad_outputs=grad_output, retain_graph=True)[0]
    print(torch.mean(grad_input,axis=0))
    #for i in range (10):
        #avg_gradient_i = gradients[:, i, :, :].mean()
        #print(average_gradients_i)
    #o.backward()
    #print(x.grad)
    #weights = o.linear1.weight.data.numpy()

    # Create a heatmap of the weights
    #sns.heatmap(weights, cmap="YlGnBu")
    #plt.xlabel("Feature 1")
    #plt.ylabel("Feature 2")
    #plt.title("Weights of FC1 Layer")
    #plt.show()
    input0 = torch.cat((torch.from_numpy(e1[:, :, 0]).float(), torch.from_numpy(e1[:, :, 1]).float()), 1)
    ex = shap.DeepExplainer(o, input0)
    shap_values = ex.shap_values(input0)
    print(np.mean(shap_values,axis=0))

    # print the JS visualization code to the notebook
    #shap.initjs()
    #color = shap_values[:, 3]
    #plt.scatter(e1[:, 3, 1], e1[:, 3, 0],c=color)
    #plt.show()
    # shap.summary_plot(shap_values, input0.numpy())
    #df_x=pd.DataFrame(input0.numpy(),columns=['cost','time','delay','emission','transshippment','pc','pt','pd','pe','ptr'])
    #shap.force_plot(ex.expected_value, shap_values[0,:],df_x.iloc[0,:])

def basenn():
    o=base()
    optimizer = torch.optim.AdamW(o.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        loss_sum = 0
        for step, (batch_x_0, batch_x_1, batch_y) in enumerate(train_dataloader):
            u1 = o(batch_x_0[:, :, 0].float())
            u2 = o(batch_x_1[:, :, 0].float())
            utility1 = u1
            utility2 = u2
            input = torch.cat((utility1.reshape(-1, 1), utility2.reshape(-1, 1)), 1)
            target = batch_y.long()
            loss = criterion(input, target)
            # take gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum = loss + loss_sum

        # print(epoch, 'nn',loss_sum/(train_n/batch_size)) #real
        if (epoch + 1) % 50 == 0:
            print("Epoch [{}/100], Loss: {:.4f}".format(epoch + 1, loss_sum / (train_n / batch_size)))

            incorrect = 0
            index = []
            c_index = []
            u1 = o(torch.from_numpy(e1[:, :, 0]).float())
            u2 = o(torch.from_numpy(e2[:, :, 0]).float())

            utility1 = u1
            utility2 = u2

            for i in range(len(e1[:, 0, 0])):
                if utility1[i] > utility2[i] and test_real_compare[i] == 1:
                    incorrect = incorrect + 1
                    index.append(test_h[i])
                elif utility1[i] < utility2[i] and test_real_compare[i] == 0:
                    incorrect = incorrect + 1
                    index.append(test_h[i])
                else:
                    c_index.append(test_h[i])

            print('accuracy', 1 - incorrect / test_n)
def dcm():

    n=len(real_compare)
    df=pd.DataFrame()
    df['CHOICE']=real_compare+1
    df['cost_1']=r1[:,0]
    df['time_1']=r1[:,1]
    df['delay_1']=r1[:,2]
    df['emission_1']=r1[:,3]
    df['trans_1']=r1[:,4]
    df['c']=x1[:, 0, 1]
    df['t']=x1[:, 1, 1]
    df['d']=x1[:, 2, 1]
    df['e']=x1[:, 3, 1]
    df['tr']=x1[:, 4, 1]
    df['A1']=np.ones(n)

    df['cost_2']=r2[:,0]
    df['time_2']=r2[:,1]
    df['delay_2']=r2[:,2]
    df['emission_2']=r2[:,3]
    df['trans_2']=r2[:,4]

    df['A2']=np.ones(n)

    database = db.Database("pl",df)
    globals().update(database.variables)

    #Create parameters to be estimated
    B_TIME = Beta('B_TIME',0,None ,None ,0)
    B_COST = Beta('B_COST',0,None ,None ,0)
    B_DELAY = Beta('B_DELAY',0,None ,None ,0)
    B_EMISSION = Beta('B_EMISSION',0,None ,None ,0)
    B_TRANS = Beta('B_TRANS',0,None ,None ,0)

    B_TIMEp = Beta('B_TIMEp',0,None ,None ,0)
    B_COSTp = Beta('B_COSTp',0,None ,None ,0)
    B_DELAYp = Beta('B_DELAYp',0,None ,None ,0)
    B_EMISSIONp = Beta('B_EMISSIONp',0,None ,None ,0)
    B_TRANSp= Beta('B_TRANSp',0,None ,None ,0)
    #Define the utility functions
    #V1 = B_COST * cost_1 + B_TIME * time_1 + B_DELAY * delay_1 + B_EMISSION * emission_1+ B_TRANS * trans_1 \
         #+ B_COSTp * c + B_TIMEp * t + B_DELAYp * d + B_EMISSIONp * e + B_TRANSp* tr
    #V2 = B_COST * cost_2 + B_TIME * time_2 + B_DELAY * delay_2 + B_EMISSION * emission_2+ B_TRANS * trans_2 \
         #+ B_COSTp * c + B_TIMEp * t + B_DELAYp * d + B_EMISSIONp * e + B_TRANSp * tr
    V1 = B_COST * cost_1 + B_TIME * time_1 + B_DELAY * delay_1 + B_EMISSION * emission_1+ B_TRANS * trans_1
    V2 = B_COST * cost_2 + B_TIME * time_2 + B_DELAY * delay_2 + B_EMISSION * emission_2+ B_TRANS * trans_2
    #Associate utility functions with alternatives and associate availability of alternatives
    V = {1: V1,
         2: V2}

    av = {1: A1,
          2: A2}

    #Define the model
    logprob = models.loglogit(V, av, CHOICE)

    #Define the Biogeme object
    biogeme  = bio.BIOGEME(database, logprob)
    biogeme.modelName = "preference_logit_estimators"

    results = biogeme.estimate()

    #Print results
    betas = results.getBetaValues()
    for k,v in betas.items():
        print(f"{k:10}=\t{v:.3g}")

    Results = results.getEstimatedParameters()
    #print(Results)

    #general statistics
    gs = results.getGeneralStatistics()

    for k,v in gs.items():
        print("{}= {}".format(k.ljust(45),v[0]))

    # validation
    n=len(test_real_compare)
    df_test=pd.DataFrame()

    df_test=pd.DataFrame()
    df_test['CHOICE']=test_real_compare+1
    df_test['cost_1']=t1[:,0]
    df_test['time_1']=t1[:,1]
    df_test['delay_1']=t1[:,2]
    df_test['emission_1']=t1[:,3]
    df_test['trans_1']=t1[:,4]
    df_test['c']=e1[:, 0, 1]
    df_test['t']=e1[:, 1, 1]
    df_test['d']=e1[:, 2, 1]
    df_test['e']=e1[:, 3, 1]
    df_test['tr']=e1[:, 4, 1]
    df_test['A1']=np.ones(n)

    df_test['cost_2']=t2[:,0]
    df_test['time_2']=t2[:,1]
    df_test['delay_2']=t2[:,2]
    df_test['emission_2']=t2[:,3]
    df_test['trans_2']=t2[:,4]
    df_test['A2']=np.ones(n)

    database_test = db.Database("pl_test",df_test)
    globals().update(database_test.variables)

    prob_1 = models.logit(V, av, 1)
    prob_2 = models.logit(V, av, 2)
    simulate ={'Prob. 1':  prob_1,
               'Prob. 2':  prob_2}
    biogeme = bio.BIOGEME(database_test, simulate)
    biogeme.modelName = "logit_test"
    betas = biogeme.freeBetaNames

    print('Extracting the following variables:')
    for k in betas:
        print('\t',k)

    results = res.bioResults(pickleFile='preference_logit_estimators.pickle')
    betaValues = results.getBetaValues ()
    simulatedValues = biogeme.simulate(betaValues)
    print(simulatedValues.head())
    prob_max = simulatedValues.idxmax(axis=1)
    prob_max = prob_max.replace({'Prob. 1': 1, 'Prob. 2': 2})

    data = {'y_Actual':    df_test['CHOICE'],
            'y_Predicted': prob_max
            }

    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])


    accuracy = np.diagonal(confusion_matrix.to_numpy()).sum()/confusion_matrix.to_numpy().sum()
    acc1 = confusion_matrix.to_numpy()[0,0]
    acc2 = confusion_matrix.to_numpy()[1, 1]
    print('Global accuracy of the model:', accuracy, acc1, acc2)

#seg()
#one()
#dcm()
basenn()

