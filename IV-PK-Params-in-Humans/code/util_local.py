import pandas as pd
from util_global import pd_col_onehotnan


def load_data(filename_data_in) :
    
    feature = dict()
    
    feature['1'] = \
    [ 'Name' ,
      'CAS #' ]
    
    feature['2a'] = \
    [ 'human VDss (L/kg)' ,
      'human CL (mL/min/kg)' ]
    
    feature['2b'] = 'fraction unbound in plasma (fu)'
    
    feature['2c'] = \
    [ 'MRT (h)' ,
      'terminal  t1/2 (h)' ]
    
    feature['3'] = \
    [ 'MW' ,
      'HBA' ,
      'HBD' ,
      'TPSA_NO' ,
      'RotBondCount' ,
      'MoKa.LogP' ,
      'MoKa.LogD7.4' ]
    
    feature['4a'] = 'moka_ionState7.4'
    
    feature['2'] = feature['2a'] + [feature['2b']] + feature['2c']
    
    feature['A'] = feature['1'] + feature['2'] + feature['3'] + [feature['4a']]
    
    dataA = pd.read_csv( filename_data_in , usecols = feature['A'] )
    cell_has_missing = dataA[ feature['2a'] + [feature['2b']] ].isna()
    row_has_missing  = cell_has_missing.any(axis=1)
    dataB1 = dataA[~row_has_missing]
    dataB2 , feature['4b'] = pd_col_onehotnan( dataB1[feature['4a']] , feature['4a'] , '_' )
    dataB = pd.concat( [ dataB1 , dataB2 ] , axis = 1 )
    feature['5'] = []
    for feat in feature['2a'] :
        addl_feat = feat + ' / ' + feature['2b']
        feature['5'].append(addl_feat)
        dataB[addl_feat] = dataB[feat] / dataB[feature['2b']]
    
    feature['B'] = feature['A'] + feature['4b'] + feature['5']
    
    return dataA , dataB , feature

