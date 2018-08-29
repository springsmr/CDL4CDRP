# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
from smote import Smote

def load_dermatology_dataset(filename):
    data=pd.read_csv(filename,delimiter=',')
    target_names_nums={'psoriasis':112,'seboreic dermatitis':61,'lichen planus':72,'pityriasis rosea':49,'cronic dermatitis':52,'pityriasis rubra pilaris':20}
    target_names=['psoriasis','seboreic dermatitis','lichen planus', 'pityriasis rosea','cronic dermatitis','pityriasis rubra pilaris']
    target_nums=[112,61,72,49,52,20]
    target_labels=[]
    if data.shape[1]==34:
        y_num=pd.read_csv('dermatologyNumric.csv',delimiter=',')['target-label-num']
        data=data.jion(y_num)
    y_num=data['target-label-num']
    for c in y_num:
        target_labels.append(target_names[int(c-1)])
    plot_data=data.copy()
    y_class=pd.DataFrame(target_labels,columns=['target-label'])
    plot_data['target-label']=y_class
    return plot_data,data

def get_category(data,k=6):
    ages_with_nan=data[' Age (linear)']
    ages=sorted(ages_with_nan.dropna().values)
    rate_list=[]
    for i in range(1,k):
        rate_list.append(i/float(k))
    rate=np.array(rate_list)
    qs=len(ages)*rate
    qs1=np.ceil(qs)
    qs2=np.floor(qs)
    qs1=qs1.astype(np.int32)
    qs2=qs2.astype(np.int32)
    qss=[]
    for i in range(len(rate)):
        qss.append((ages[qs1[i]]+ages[qs2[i]])/2)
    def change2category(x):
        if not np.isnan(x):
            c=0
            while(c<len(qss) and x>qss[c]):
                c+=1
            return c
        return x
    ages_c=map(change2category,ages_with_nan.values)
    new_ages=pd.Series(ages_c)
    data[' Age (linear)']=new_ages
    return data

def plot_data(plot_data):
    plt.figure(figsize=(15,8))
    parallel_coordinates(plot_data,'target-label')
    plt.show()
    return None
def get_smote_sampledata(data):
    x=data.iloc[:,:-1].values
    y=data.iloc[:,-1].values #数字类别
    target_nums=[112,61,72,49,52,20]
    mx=target_nums[0]
    x_sample=x[y==1]
    y_sample=np.ones((len(x_sample),1))
    for i in range(1,6):
        srate=int(np.floor(mx/target_nums[i]))
        print srate
        st=Smote(sampling_rate=srate, k=6)
        xo=(st.fit(x[y==(i+1)]))
        x_sample=np.concatenate((x_sample,xo), axis=0)
        y_sample=np.concatenate((y_sample,(i+1)*np.ones((len(xo),1))), axis=0)
    data_sample=np.column_stack((x_sample,y_sample))
    return x_sample,y_sample
