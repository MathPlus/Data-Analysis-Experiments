import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
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


def plotloglog_lenses_pair( x , y , x_label , y_label , fig_titlebase , fig_filename ) :
    ( r , r_pval ) = pearsonr( np.log10(x) , np.log10(y) )
    fig_title = '{titlebase}; loglogR = {loglogR:.1f}'.format( titlebase = fig_titlebase , loglogR = r )
    fig , ax = plt.subplots()
    plt.loglog( x , y , 'r.' )
    plt.xlabel( x_label )
    plt.ylabel( y_label )
    plt.title( fig_title )
    plt.savefig( fig_filename )
    plt.show()
    plt.close()