import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from util_global import pd_col_onehotnan , round_up , round_dw
import kmapper as km
import networkx as nx


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


def get_node_idx_pair(node_id_str) :
    _ , _ , node_id_substr = node_id_str.partition('cube')
    patch_idx_str , _ , node_idx_str = node_id_substr.partition('_cluster')
    patch_idx = int(patch_idx_str)
    node_idx = int(node_idx_str)
    return patch_idx , node_idx


def get_tda_remodel(tda_model) :
    
    node_id_str_list = list( tda_model['nodes'].keys() )
    
    node_id_data_list_ = [ ( get_node_idx_pair(node_id_str) , node_id_str )
                           for node_id_str in node_id_str_list ]
    
    node_id_data_list = sorted( node_id_data_list_ , key = lambda node_id_data : node_id_data[0] )
    
    node_data_list = [ { 'patch_idx'              : node_id_data[0][0] ,
                         'cluster_idx'            : node_id_data[0][1] ,
                         'node_size'              : None , #len(tda_model['nodes'][ node_id_data[1]]) ,
                         'node_size/dataset_size' : None ,
                         'row_idx_list'           : sorted( tda_model['nodes'][ node_id_data[1]] ) }
                       for node_id_data in node_id_data_list ]
    
    nber_nodes = len(node_data_list)
    
    edge_list_ = [ item for item in tda_model['simplices'] if len(item) == 2 ]
    nber_edges = len(edge_list_)
    edge_list = [None] * nber_edges
    
    for i in range(nber_edges) :
        nodeA_idx_list = [ idx for idx in range(nber_nodes) if node_id_data_list[idx][1] == edge_list_[i][0] ]
        nodeB_idx_list = [ idx for idx in range(nber_nodes) if node_id_data_list[idx][1] == edge_list_[i][1] ]
        assert( len(nodeA_idx_list) ==  1 )
        assert( len(nodeB_idx_list) ==  1 )
        assert( nodeA_idx_list[0] != nodeB_idx_list[0] )
        if nodeA_idx_list[0] < nodeB_idx_list[0] :
            nodeA_idx = nodeA_idx_list[0]
            nodeB_idx = nodeB_idx_list[0]
        else :
            nodeA_idx = nodeB_idx_list[0]
            nodeB_idx = nodeA_idx_list[0]
        edge_list[i] = ( nodeA_idx , nodeB_idx )
    
    edge_data_list = [ { 'nodeA_idx'              : edge[0] ,
                         'nodeB_idx'              : edge[1] ,
                         'edge_size'              : None ,
                         'edge_size/dataset_size' : None ,
                         'edge_size/nodeA_size'   : None ,
                         'edge_size/nodeB_size'   : None ,
                         'row_idx_list'           : None ,
                         'nodeA_ovlp_idx_list'    : None ,
                         'nodeB_ovlp_idx_list'    : None }
                       for edge in edge_list ]
    
    return node_data_list , edge_data_list


# def make_tda_graph(tda_model) :
#     tda_graph = nx.Graph()
    