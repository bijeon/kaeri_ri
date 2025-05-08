import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from tqdm import tqdm

import psutil
import ray
import sys

warnings.filterwarnings('ignore')

def get_hist(eng,data):
    hist = np.zeros_like(eng)
    
    for i in range(len(eng)):
        for d in data:
            if abs(d-eng[i]) <= 1e-5:
                hist[i] += 1
    return hist

def spt_sampling(eng,data):

    # 핵종의 갯수를 정함, 1개, 2개, 3개 뽑는 비율,
    num_ri = np.random.choice([1,2,3], size=1, p=[6/11,3/11,2/11])# Number of RIs

    # 핵종의 비율을 정함, 최소 비율은 0.1
    ratio_ri = np.random.random(num_ri)
    np.clip(ratio_ri, a_min = 0.1, a_max=1, out = ratio_ri)

    ratio_ri = ratio_ri/np.sum(ratio_ri)
    ratios = np.zeros(3)

    ratios[:int(num_ri)] = ratio_ri
    np.random.shuffle(ratios)
    # 정해진 비율대로 스펙트럼을 더하여 새로운 스펙트럼을 생성
    new_pdf = np.sum(data*ratios,axis=1)

    # 샘플링 할 카운트 수를 정함
    num_counts = np.random.randint(low=3000, high=5000)

    # 카운트 수 대로 샘플링을 수행하여 스펙트럼을 구함
    events = np.random.choice(eng,size=num_counts,replace=True,p=new_pdf)
    spt = get_hist(eng,events)

    return np.append(ratios,spt)


data = pd.read_csv('./NaI_spectra2.csv').to_numpy()
eng = data[1::2,0]
data = np.sum(data[1:,1::2].reshape(-1,2,3),axis=1)
data = data/np.sum(data,axis=0)


plt.plot(eng,data[:,0],label='Na22')
plt.plot(eng,data[:,1],label='Co60')
plt.plot(eng,data[:,2],label='Cs137')
plt.yscale('log')
plt.xlabel('Energy (MeV)')
plt.ylabel('Counts (A.U.)')
plt.legend()

events = np.random.choice(eng,size=1000,replace=True,p=data[:,0]/np.sum(data[:,0]))

spt = get_hist(eng,events)
plt.plot(eng,spt)


# # for 문
# spectra = []
# # for i in tqdm(range(5000)):
# for i in range(10):
#     results = spt_sampling(spt,data)
#     spectra.append(results)
# spectra=np.array(spectra)

## Ray 이용
num_logical_cpus = psutil.cpu_count()
ray.init(num_cpus= num_logical_cpus)

@ray.remote
def ray_spt_sampling(eng,data):
    # 핵종의 갯수를 정함, 1개, 2개, 3개 뽑는 비율,
    num_ri = np.random.choice([1,2,3], size=1, p=[6/11,3/11,2/11])# Number of RIs

    # 핵종의 비율을 정함, 최소 비율은 0.1
    ratio_ri = np.random.random(num_ri)
    np.clip(ratio_ri, a_min = 0.1, a_max=1, out = ratio_ri)

    ratio_ri = ratio_ri/np.sum(ratio_ri)
    ratios = np.zeros(3)

    ratios[:int(num_ri)] = ratio_ri
    np.random.shuffle(ratios)
    # 정해진 비율대로 스펙트럼을 더하여 새로운 스펙트럼을 생성
    new_pdf = np.sum(data*ratios,axis=1)

    # 샘플링 할 카운트 수를 정함
    num_counts = np.random.randint(low=3000, high=5000)

    # 카운트 수 대로 샘플링을 수행하여 스펙트럼을 구함
    events = np.random.choice(eng,size=num_counts,replace=True,p=new_pdf)
    spt = get_hist(eng,events)

    return np.append(ratios,spt)

Energy = ray.put(eng)
Data = ray.put(data)
result_ids = [ray_spt_sampling.remote(Energy,Data) for x in range(5000)]
results = ray.get(result_ids)

spectra = np.array(results)

ray.shutdown()

np.savetxt('./DATASET.csv',spectra,delimiter=",")
