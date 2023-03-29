import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import biogeme.biogeme as bio
import biogeme.database as db
import biogeme.models as models
import biogeme.results as res
from Benchmark import BL
from biogeme.expressions import Beta, DefineVariable

class prediction_validation():
    def __init__(self, real_compare,model):
        self.criterion = nn.CrossEntropyLoss()
        self.model = model
        self.real_choice =real_compare

    def predict_choice_dcm(self,df_test,pickleFile):
        b = BL(df_test)
        predicted_choice = b.validate(pickleFile)
        return predicted_choice

    def predict_choice_nn(self,r,test_route_0,test_route_1):
        test_state_0 = torch.from_numpy(test_route_0).float()
        test_state_1 = torch.from_numpy(test_route_1).float()
        test_reward_0 = r(test_state_0).detach().numpy()
        test_reward_1 = r(test_state_1).detach().numpy()
        reward = np.zeros((2, len(test_reward_0)))
        reward[0,:] = list(test_reward_0)
        reward[1,:] = list(test_reward_1)
        predicted_choice = np.argmax(reward, axis=0)

        return predicted_choice


    def validate_dcm(self,df_test,pickleFile):
        # loglikelihood
        # accuracy
        predicted_choice=self.predict_choice_dcm(df_test,pickleFile)

        data = {'y_Actual': self.real_choice,
                'y_Predicted': predicted_choice
                }

        data  = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
        confusion_matrix = pd.crosstab(data['y_Actual'], data['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        accuracy = np.diagonal(confusion_matrix.to_numpy()).sum() / confusion_matrix.to_numpy().sum()
        #log_likelihood
        return accuracy

    def validate_nn(self,r,test_route_0,test_route_1):

        predicted_choice=self.predict_choice_nn(r,test_route_0,test_route_1)

        data = {'y_Actual': self.real_choice,
                'y_Predicted': predicted_choice
                }

        data  = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
        confusion_matrix = pd.crosstab(data['y_Actual'], data['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        accuracy = np.diagonal(confusion_matrix.to_numpy()).sum() / confusion_matrix.to_numpy().sum()
        #log_likelihood
        return accuracy

        '''
        test_estimated_r_0 = torch.reshape(test_estimated_r_0, (len(test_route_0[:, 0]), -1))  # value for route_0
        test_estimated_r_1 = torch.reshape(test_estimated_r_1, (len(test_route_0[:, 0]), -1))  # value for route_1

        input = torch.cat((test_estimated_r_0, test_estimated_r_1), 1)
        target = torch.from_numpy(test_real_compare).long()
        log_likelihood = -self.criterion(input, target)
        return log_likelihood.detach().numpy(), accuracy
        '''