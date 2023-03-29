import pandas as pd
import numpy as np
import par
def variable_description():
    '''
    :return: table: mean, Std. Dev.,min,25,50,75,max,count
    '''
    df=pd.read_csv('D:\Research\pypy\RL\RL_new\ALNS_PL\ALNS_PL_3.28\learning\linear/train_data.csv')
    data=df.values
    douput=np.zeros((5,8))
    output_columns=['Mean','Std','Min','25%','50%','75%','Max','Count']
    for i in range (1,5):
        douput[i, 0]= np.mean(data[:,i])
        douput[i, 1] = np.percentile(data[:,i], 25)
        douput[i, 2] = np.min(data[:, i])
        douput[i, 3] = np.percentile(data[:,i], 25)
        douput[i, 4] = np.percentile(data[:,i], 50)
        douput[i, 5] = np.percentile(data[:,i], 75)
        douput[i, 6] = np.max(data[:, i])
        douput[i, 7] = len(data[:, i])

    douput = pd.DataFrame(douput,columns=output_columns)
    douput['Variables'] = ['Cost', 'Time', 'Delay', 'Emission', 'Transshipment']
    douput.to_csv('D:\Research\pypy\RL\RL_new\ALNS_PL\ALNS_PL_3.28\learning\linear/variable.csv')

def choice_distribution():
    '''
    :return: table: A1,A2,A3: y=0, y=1, the percentage
    '''
    df_1 = pd.read_csv('comparisons_linear.csv')
    df_2 = pd.read_csv('comparisons_piecewise.csv')
    df_3 = pd.read_csv('comparisons_nonlinear.csv')
    douput = pd.DataFrame()
    douput['model']=['S1','S2','S3']
    douput['y=0'] = [1 - df_1['CHOICE'].mean(),1 - df_2['CHOICE'].mean(),1 - df_3['CHOICE'].mean()]
    douput['y=1'] = [df_1['CHOICE'].mean(),df_2['CHOICE'].mean(),df_3['CHOICE'].mean()]
    douput.to_csv(par.base_file+'/output/choice_distribution.csv')

    doutput2=pd.DataFrame(index=0)
    doutput2['model'] = ['S1-S2', 'S2-S3', 'S1-S3']
    doutput2['changed choices'] = [change_choice(df_1['CHOICE'],df_2['CHOICE']),
                                  change_choice(df_2['CHOICE'],df_3['CHOICE']),
                                  change_choice(df_1['CHOICE'],df_3['CHOICE'])]
    doutput2.to_csv(par.base_file + '/output/choice_changed.csv')

def change_choice (a, b):
    '''
    model: synthetic data
    :return: table: A1-A2, A2-A3, A1-A3, the percentage of choices changed
    '''
    change=0
    for i in range (len(a)):
        if a[i]!=b[i]:
            change=change+1
    return change

variable_description()