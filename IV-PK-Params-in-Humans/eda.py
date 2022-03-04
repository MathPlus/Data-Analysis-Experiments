import pandas as pd
import matplotlib.pyplot as plt
from util import distance_Thompson_base10

filename_data_in = "Data-in/IV-PK-Params-in-Humans.csv"

feature1 = \
[ "Name" ,
  "CAS #" ]

feature2a = \
[ "human VDss (L/kg)" ,
  "human CL (mL/min/kg)" ]

feature2b = \
[ "fraction unbound in plasma (fu)" ,
  "MRT (h)" ,
  "terminal  t1/2 (h)" ]

feature3 = \
[ "MW" ,
  "HBA" ,
  "HBD" ,
  "TPSA_NO" ,
  "RotBondCount" ,
  "moka_ionState7.4" ,
  "MoKa.LogP" ,
  "MoKa.LogD7.4" ]

feature2 = feature2a + feature2b
feature = feature1 + feature2 + feature3

data_all = pd.read_csv( filename_data_in , usecols = feature )

cell_has_missing2a = data_all[feature2a].isna()
row_has_missing2a = cell_has_missing2a.any(axis=1)

data = data_all[~row_has_missing2a]

cell_has_missing2 = data[feature2].isna()
row_has_missing2 = cell_has_missing2.any(axis=1)

tda_data = data[feature2].to_numpy()

fig , ax = plt.subplots()
plt.loglog( tda_data[:,0] , tda_data[:,1] , 'r.' )
plt.xlabel( feature2[0] )
plt.ylabel( feature2[1] )

n = tda_data.shape[0]
N = n * ( n - 1 ) // 2
distce = [None] * N
k = -1
for i in range(n) :
    for j in range(i+1,n) :
        k = k + 1
        distce[k] = distance_Thompson_base10( tda_data[i,:] ,
                                              tda_data[j,:] ,
                                              "Linf" )

fig , ax = plt.subplots()
plt.hist( distce , bins = 50 , density = True , facecolor = 'g' )
plt.xlabel( "Pairwise distances" )
plt.ylabel( "Density" )
