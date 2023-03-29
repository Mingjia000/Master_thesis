import pandas  as pd
import numpy as np
import biogeme.biogeme as bio
import biogeme.database as db
import biogeme.models as models
import biogeme.results as res
from biogeme.expressions import Beta, DefineVariable
import pickle
class BL():
    def __init__(self, df):
        #self.criterion = nn.CrossEntropyLoss()
        self.df = df
    def train(self):
        database = db.Database("pl", self.df)
        globals().update(database.variables)
        # Create parameters to be estimated
        B_TIME = Beta('B_TIME', 0, None, None, 0)
        B_COST = Beta('B_COST', 0, None, None, 0)
        B_DELAY = Beta('B_DELAY', 0, None, None, 0)
        B_EMISSION = Beta('B_EMISSION', 0, None, None, 0)
        B_TRANS = Beta('B_TRANS', 0, None, None, 0)

        # Define the utility functions
        V1 = B_COST * cost_1 + B_TIME * time_1 + B_DELAY * delay_1 + B_EMISSION * emission_1 + B_TRANS * trans_1
        V2 = B_COST * cost_2 + B_TIME * time_2 + B_DELAY * delay_2 + B_EMISSION * emission_2 + B_TRANS * trans_2

        # Associate utility functions with alternatives and associate availability of alternatives
        V = {0: V1,1: V2}

        av = {0: A1,1: A2}

        # Define the model
        logprob = models.loglogit(V, av, CHOICE)

        # Define the Biogeme object
        biogeme = bio.BIOGEME(database, logprob)
        biogeme.modelName = 'BL'

        # Estimate the model
        biogeme.generateHtml = False
        biogeme.generatePickle = False

        results = biogeme.estimate()
        #results = res.bioResults(pickleFile='preference_logit_estimators.pickle')
        # Print results
        betas = results.getBetaValues()
        #for k, v in betas.items():
            #print(f"{k:10}=\t{v:.3g}")
        #Results = results.getEstimatedParameters()
        return results,betas

    def validate(self,pickleFile):
        database_test = db.Database("pl_test", self.df)
        globals().update(database_test.variables)

        # Create parameters to be estimated
        B_TIME = Beta('B_TIME', 0, None, None, 0)
        B_COST = Beta('B_COST', 0, None, None, 0)
        B_DELAY = Beta('B_DELAY', 0, None, None, 0)
        B_EMISSION = Beta('B_EMISSION', 0, None, None, 0)
        B_TRANS = Beta('B_TRANS', 0, None, None, 0)

        V1 = B_COST * cost_1 + B_TIME * time_1 + B_DELAY * delay_1 + B_EMISSION * emission_1 + B_TRANS * trans_1
        V2 = B_COST * cost_2 + B_TIME * time_2 + B_DELAY * delay_2 + B_EMISSION * emission_2 + B_TRANS * trans_2

        V = {0: V1, 1: V2}
        av = {0: A1, 1: A2}
        prob_1 = models.logit(V, av, 0)
        prob_2 = models.logit(V, av, 1)
        simulate = {'Prob. 1': prob_1,
                    'Prob. 2': prob_2}

        biogeme = bio.BIOGEME(database_test, simulate)
        biogeme.modelName = "logit_test"

        # betas = biogeme.freeBetaNames
        #results = res.bioResults(pickleFile=pickleFile)
        results=pickle.load( open( pickleFile, "rb" ))
        betaValues = results.getBetaValues()
        simulatedValues = biogeme.simulate(betaValues)
        prob_max = simulatedValues.idxmax(axis=1)
        predicted_choice = prob_max.replace({'Prob. 1': 0, 'Prob. 2': 1})
        return predicted_choice
