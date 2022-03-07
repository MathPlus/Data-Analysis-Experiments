import pandas as pd
import matplotlib.pyplot as plt
from util import distance_Thompson_base10 , pd_col_onehotnan

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

feature3a = \
[ "MW" ,
  "HBA" ,
  "HBD" ,
  "TPSA_NO" ,
  "RotBondCount" ,
  "MoKa.LogP" ,
  "MoKa.LogD7.4" ]

feature3b = "moka_ionState7.4"

feature2 = feature2a + feature2b
featureA = feature1 + feature2 + feature3a + [feature3b]

dataA = pd.read_csv( filename_data_in , usecols = featureA + [feature3b] )

dataOneHot_moka_ionState74 = pd_col_onehotnan( dataA[feature3b] , feature3b , "_" )

dataB = pd.concat( [ dataA[featureA] , dataOneHot_moka_ionState74 ] , axis = 1 )

featureB = featureA + list( dataOneHot_moka_ionState74.columns )

cell_has_missing2a = dataB[feature2a].isna()
row_has_missing2a = cell_has_missing2a.any(axis=1)

data = dataB[~row_has_missing2a]

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
