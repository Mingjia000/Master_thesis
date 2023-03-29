import numpy as np
import pandas as pd

def gen_demand(r_num):
    '''
    :param r_num: number of request
    :return: dataframe for requests
    '''
    r=pd.DataFrame(columns=['p','d','ap','bp','ad','bd','qr'])
    r['p']=np.random.randint(low=1, high=3, size=r_num)
    r['d']=np.random.randint(low=4, high=11, size=r_num)
    r['ap']=np.random.randint(low=1, high=120, size=r_num)
    r['bp']=r['ap']+ np.random.randint(low=20, high=80, size=r_num)
    r['ad']=r['ap']
    r['bd']=r['bp']
    r['qr']=np.random.randint(low=10, high=30, size=r_num)
    r.reset_index(drop=True, inplace=True)
    return r
