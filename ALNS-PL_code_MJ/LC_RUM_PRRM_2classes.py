########################################
#
# @file LC_RUM_PRRM_2classes.py
# @author: Sander van Cranenburgh
# @date: 26/07/2016
#
#######################################

from biogeme import *
from headers import *
from loglikelihood import *
from statistics import *

#Parameters to be estimated
# Arguments:
#   - 1  Name for report; Typically, the same as the variable.
#   - 2  Starting value.
#   - 3  Lower bound.
#   - 4  Upper bound.
#   - 5  0: estimate the parameter, 1: keep it fixed.
#

# Parameters Class 1 RUM
B_FSG_1 = Beta('FSG_1',0,-100,100,0)
B_FSO_1 = Beta('FSO_1',0,-100,100,0)
B_TT_1  = Beta('TT_1',0,-100,100,0)

# Parameters Class 2 PRRM
B_FSG_2 = Beta('FSG_2',0,-100,100,0)
B_FSO_2 = Beta('FSO_2',0,-100,100,0)
B_TT_2  = Beta('TT_2', 0,-100,100,0)

# Class membership parameters
s = Beta('s',1,0,1,0)

# Define PRRM variables
FSG1_sc = DefineVariable('FSG1_sc', ( 1 / 1000 ) * FSG1 )
FSG2_sc = DefineVariable('FSG2_sc', ( 1 / 1000 ) * FSG2 )
FSG3_sc = DefineVariable('FSG3_sc', ( 1 / 1000 ) * FSG3 )
FSG4_sc = DefineVariable('FSG4_sc', ( 1 / 1000 ) * FSG4 )
FSG5_sc = DefineVariable('FSG5_sc', ( 1 / 1000 ) * FSG5 )

FSO1_sc = DefineVariable('FSO1_sc', ( 1 / 1000 ) * FSO1 )
FSO2_sc = DefineVariable('FSO2_sc', ( 1 / 1000 ) * FSO2 )
FSO3_sc = DefineVariable('FSO3_sc', ( 1 / 1000 ) * FSO3 )
FSO4_sc = DefineVariable('FSO4_sc', ( 1 / 1000 ) * FSO4 )
FSO5_sc = DefineVariable('FSO5_sc', ( 1 / 1000 ) * FSO5 )

TT1_sc = DefineVariable('TT1_sc', ( 1 / 100 ) * TT1 )
TT2_sc = DefineVariable('TT2_sc', ( 1 / 100 ) * TT2 )
TT3_sc = DefineVariable('TT3_sc', ( 1 / 100 ) * TT3 )
TT4_sc = DefineVariable('TT4_sc', ( 1 / 100 ) * TT4 )
TT5_sc = DefineVariable('TT5_sc', ( 1 / 100 ) * TT5 )

# Compute P-RRM Atrribute levels
X_FSG1 = DefineVariable('X_FSG1', ( max( 0 , FSG2_sc - FSG1_sc ) + max( 0 , FSG3_sc - FSG1_sc ) + max( 0 , FSG4_sc - FSG1_sc ) + max( 0 , FSG5_sc - FSG1_sc ) ))
X_FSG2 = DefineVariable('X_FSG2', ( max( 0 , FSG1_sc - FSG2_sc ) + max( 0 , FSG3_sc - FSG2_sc ) + max( 0 , FSG4_sc - FSG2_sc ) + max( 0 , FSG5_sc - FSG2_sc ) ))
X_FSG3 = DefineVariable('X_FSG3', ( max( 0 , FSG1_sc - FSG3_sc ) + max( 0 , FSG2_sc - FSG3_sc ) + max( 0 , FSG4_sc - FSG3_sc ) + max( 0 , FSG5_sc - FSG3_sc ) ))
X_FSG4 = DefineVariable('X_FSG4', ( max( 0 , FSG1_sc - FSG4_sc ) + max( 0 , FSG2_sc - FSG4_sc ) + max( 0 , FSG3_sc - FSG4_sc ) + max( 0 , FSG5_sc - FSG4_sc ) ))
X_FSG5 = DefineVariable('X_FSG5', ( max( 0 , FSG1_sc - FSG5_sc ) + max( 0 , FSG2_sc - FSG5_sc ) + max( 0 , FSG3_sc - FSG5_sc ) + max( 0 , FSG4_sc - FSG5_sc ) ))

X_FSO1 = DefineVariable('X_FSO1', ( max( 0 , FSO2_sc - FSO1_sc ) + max( 0 , FSO3_sc - FSO1_sc ) + max( 0 , FSO4_sc - FSO1_sc ) + max( 0 , FSO5_sc - FSO1_sc ) ))
X_FSO2 = DefineVariable('X_FSO2', ( max( 0 , FSO1_sc - FSO2_sc ) + max( 0 , FSO3_sc - FSO2_sc ) + max( 0 , FSO4_sc - FSO2_sc ) + max( 0 , FSO5_sc - FSO2_sc ) ))
X_FSO3 = DefineVariable('X_FSO3', ( max( 0 , FSO1_sc - FSO3_sc ) + max( 0 , FSO2_sc - FSO3_sc ) + max( 0 , FSO4_sc - FSO3_sc ) + max( 0 , FSO5_sc - FSO3_sc ) ))
X_FSO4 = DefineVariable('X_FSO4', ( max( 0 , FSO1_sc - FSO4_sc ) + max( 0 , FSO2_sc - FSO4_sc ) + max( 0 , FSO3_sc - FSO4_sc ) + max( 0 , FSO5_sc - FSO4_sc ) ))
X_FSO5 = DefineVariable('X_FSO5', ( max( 0 , FSO1_sc - FSO5_sc ) + max( 0 , FSO2_sc - FSO5_sc ) + max( 0 , FSO3_sc - FSO5_sc ) + max( 0 , FSO4_sc - FSO5_sc ) ))

X_TT1 = DefineVariable('X_TT1', ( min( 0 , TT2_sc - TT1_sc ) + min( 0 , TT3_sc - TT1_sc ) + min( 0 , TT4_sc - TT1_sc ) + min( 0 , TT5_sc - TT1_sc ) ))
X_TT2 = DefineVariable('X_TT2', ( min( 0 , TT1_sc - TT2_sc ) + min( 0 , TT3_sc - TT2_sc ) + min( 0 , TT4_sc - TT2_sc ) + min( 0 , TT5_sc - TT2_sc ) ))
X_TT3 = DefineVariable('X_TT3', ( min( 0 , TT1_sc - TT3_sc ) + min( 0 , TT2_sc - TT3_sc ) + min( 0 , TT4_sc - TT3_sc ) + min( 0 , TT5_sc - TT3_sc ) ))
X_TT4 = DefineVariable('X_TT4', ( min( 0 , TT1_sc - TT4_sc ) + min( 0 , TT2_sc - TT4_sc ) + min( 0 , TT3_sc - TT4_sc ) + min( 0 , TT5_sc - TT4_sc ) ))
X_TT5 = DefineVariable('X_TT5', ( min( 0 , TT1_sc - TT5_sc ) + min( 0 , TT2_sc - TT5_sc ) + min( 0 , TT3_sc - TT5_sc ) + min( 0 , TT4_sc - TT5_sc ) ))

# Utility / regret functions
# RUM class 
V1_1 = B_FSG_1 * FSG1_sc + B_FSO_1 * FSO1_sc + B_TT_1 * TT1_sc
V2_1 = B_FSG_1 * FSG2_sc + B_FSO_1 * FSO2_sc + B_TT_1 * TT2_sc
V3_1 = B_FSG_1 * FSG3_sc + B_FSO_1 * FSO3_sc + B_TT_1 * TT3_sc
V4_1 = B_FSG_1 * FSG4_sc + B_FSO_1 * FSO4_sc + B_TT_1 * TT4_sc
V5_1 = B_FSG_1 * FSG5_sc + B_FSO_1 * FSO5_sc + B_TT_1 * TT5_sc

# Associate utility functions with the numbering of alternatives
V1 = {1: V1_1,
     2: V2_1,
     3: V3_1,
     4: V4_1,
     5: V5_1}

# PRRM Class
R1_2 = B_FSG_2 * X_FSG1 + B_FSO_2 * X_FSO1 + B_TT_2 * X_TT1 
R2_2 = B_FSG_2 * X_FSG2 + B_FSO_2 * X_FSO2 + B_TT_2 * X_TT2 
R3_2 = B_FSG_2 * X_FSG3 + B_FSO_2 * X_FSO3 + B_TT_2 * X_TT3 
R4_2 = B_FSG_2 * X_FSG4 + B_FSO_2 * X_FSO4 + B_TT_2 * X_TT4 
R5_2 = B_FSG_2 * X_FSG5 + B_FSO_2 * X_FSO5 + B_TT_2 * X_TT5 

# Associate regret functions with the numbering of alternatives
R2 = {1: -R1_2,
     2: -R2_2,
     3: -R3_2,
     4: -R4_2,
     5: -R5_2}


# Associate the availability conditions with the alternatives
one =  DefineVariable('one',1)

av = {1: one,
      2: one,
      3: one,
      4: one,
      5: one}

# Class membership model
probClass1 = 1 - s
probClass2 = s

# The choice model is a logit, with availability conditions
prob1 = bioLogit(V1,av,CHOICE)
prob2 = bioLogit(R2,av,CHOICE)

# Iterator on individuals, that is on groups of rows.
metaIterator('personIter','__dataFile__','panelObsIter','ID')

# For each item of personIter, iterates on the rows of the group. 
rowIterator('panelObsIter','personIter')

#Conditional probability for the sequence of choices of an individual
ProbIndiv_1 = Prod(prob1,'panelObsIter')
ProbIndiv_2 = Prod(prob2,'panelObsIter')

# Define the likelihood function for the estimation
BIOGEME_OBJECT.ESTIMATE = Sum(log(probClass1 * ProbIndiv_1 + probClass2 * ProbIndiv_2),'personIter')

# Statistics
BIOGEME_OBJECT.PARAMETERS['optimizationAlgorithm'] = "BIO"
BIOGEME_OBJECT.PARAMETERS['numberOfThreads'] = "4"
BIOGEME_OBJECT.STATISTICS['Number of individuals'] = Sum(1,'personIter')
