import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from util_global import pd_col_onehotnan , round_up , round_dw
import kmapper as km


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


def make_tda_covering_scheme( tda_lens , precfg_tda_covering_scheme , verbo_lvl ) :
    
    tda_intvls_lowerbound0 = round_dw( min(tda_lens[:,0]) , precfg_tda_covering_scheme['lens_bound_rounding0'] )
    tda_intvls_upperbound0 = round_up( max(tda_lens[:,0]) , precfg_tda_covering_scheme['lens_bound_rounding0'] )
    tda_intvls_lowerbound1 = round_dw( min(tda_lens[:,1]) , precfg_tda_covering_scheme['lens_bound_rounding1'] )
    tda_intvls_upperbound1 = round_up( max(tda_lens[:,1]) , precfg_tda_covering_scheme['lens_bound_rounding1'] )
    
    cfg_tda_covering_scheme = dict()
    
    cfg_tda_covering_scheme['bound'] = np.array( [ [ tda_intvls_lowerbound0 , tda_intvls_upperbound0 ] ,
                                                   [ tda_intvls_lowerbound1 , tda_intvls_upperbound1 ] ] )
    
    cfg_tda_covering_scheme['count'] = [ precfg_tda_covering_scheme['intvls_count0'] ,
                                         precfg_tda_covering_scheme['intvls_count1'] ]
    
    cfg_tda_covering_scheme['overlap'] = [ precfg_tda_covering_scheme['intvls_overlap0'] ,
                                           precfg_tda_covering_scheme['intvls_overlap1'] ]
    
    tda_covering_scheme = km.Cover( limits       = cfg_tda_covering_scheme['bound'] ,
                                    n_cubes      = cfg_tda_covering_scheme['count'] ,
                                    perc_overlap = cfg_tda_covering_scheme['overlap'] ,
                                    verbose      = verbo_lvl )

    return tda_covering_scheme