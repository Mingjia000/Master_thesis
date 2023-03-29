import numpy as np
from True_preference import shipper_reward
import pandas  as pd


n=2000
df=pd.DataFrame()
df['CHOICE']=np.random.randint(2, size=n)+1
df['cost_1']=np.random.rand(n)
df['time_1']=np.random.rand(n)
df['delay_1']=np.random.rand(n)
df['emission_1']=np.random.rand(n)
df['trans_1']=np.random.rand(n)
df['A1']=np.ones(n)

df['cost_2']=np.random.rand(n)
df['time_2']=np.random.rand(n)
df['delay_2']=np.random.rand(n)
df['emission_2']=np.random.rand(n)
df['trans_2']=np.random.rand(n)
df['A2']=np.ones(n)

database = db.Database("pl",df)
globals().update(database.variables)

#Create parameters to be estimated
B_TIME = Beta('B_TIME',0,None ,None ,0)
B_COST = Beta('B_COST',0,None ,None ,0)
B_DELAY = Beta('B_DELAY',0,None ,None ,0)
B_EMISSION = Beta('B_EMISSION',0,None ,None ,0)
B_TRANS = Beta('B_TRANS',0,None ,None ,0)

#Define the utility functions
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

#Estimate the model
biogeme.generateHtml = False
biogeme.generatePickle = True

results = biogeme.estimate()

#Print results
betas = results.getBetaValues()
for k,v in betas.items():
    print(f"{k:10}=\t{v:.3g}")

Results = results.getEstimatedParameters()
print(Results)

#general statistics
gs = results.getGeneralStatistics()

for k,v in gs.items():
    print("{}= {}".format(k.ljust(45),v[0]))

# validation
n=2000
df_test=pd.DataFrame()
df_test=df
'''
'''
df_test['CHOICE']=np.random.randint(2, size=n)+1
df_test['cost_1']=np.random.rand(n)
df_test['time_1']=np.random.rand(n)
df_test['delay_1']=np.random.rand(n)
df_test['emission_1']=np.random.rand(n)
df_test['trans_1']=np.random.rand(n)
df_test['A1']=np.ones(n)

df_test['cost_2']=np.random.rand(n)
df_test['time_2']=np.random.rand(n)
df_test['delay_2']=np.random.rand(n)
df_test['emission_2']=np.random.rand(n)
df_test['trans_2']=np.random.rand(n)
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
print('Global accuracy of the model:', accuracy)