"""
========================================================================================================
Trend Analysis of a Database of Intravenous Pharmacokinetic Parameters in Humans for 1352 Drug Compounds
========================================================================================================
"""


import numpy as np
import kmapper as km
from util_global import weblink_fmt , distance_Thompson_base10
from util_local import load_data ,\
                       make_tda_covering_scheme ,\
                       parse_tda_modelKM
import sklearn
import pickle


verbo_lvl = 0


publication_url = 'https://dmd.aspetjournals.org/content/46/11/1466'
publication_title = 'Trend Analysis of a Database of Intravenous Pharmacokinetic Parameters in Humans for 1352 Drug Compounds'
publication_weblink = weblink_fmt.format( url = publication_url , descr = publication_title )

title_tda_model    = 'A topological model of the dataset studied in {publ_weblink}'.format( publ_weblink = '<i>' + publication_weblink +'</i>' )
filename_data_in   = '../data/IV-PK-Params-in-Humans.csv'
filename_data_out  = '../TDA_models/IV-PK-Params-in-Humans.pckl'
filename_tda_model = '../TDA_models/IV-PK-Params-in-Humans.html'

_ , data , feature = load_data(filename_data_in)

tda_data_hdr = feature['2']
tda_data = np.nan_to_num( data[tda_data_hdr].to_numpy() , nan = 0.0 )

tda_metric = lambda x , y : distance_Thompson_base10( x , y , 'Linf' )

tda_clusterer = sklearn.cluster.DBSCAN( eps           = 0.5 ,
                                        min_samples   = 5 ,
                                        metric        = tda_metric ,
                                        metric_params = None ,
                                        algorithm     = 'auto' ,
                                        leaf_size     = 30 ,
                                        p             = None ,
                                        n_jobs        = None )

tda_lens_data_hdr = feature['5']
tda_lens_data = np.log2( data[tda_lens_data_hdr].to_numpy() )
# min(tda_lens_data[:,0]) == -3.77595972578207
# max(tda_lens_data[:,0]) == 18.194602975157967
# min(tda_lens_data[:,1]) == -4.832890014164741
# max(tda_lens_data[:,1]) == 15.189531985610547
cfg_tda_covering_scheme = dict()
cfg_tda_covering_scheme['lens_bound_rounding0'] = 0.5
cfg_tda_covering_scheme['lens_bound_rounding1'] = 0.5
cfg_tda_covering_scheme['intvls_count0'] = 8
cfg_tda_covering_scheme['intvls_count1'] = 8
cfg_tda_covering_scheme['intvls_overlap0'] = 0.4
cfg_tda_covering_scheme['intvls_overlap1'] = 0.4

tda_covering_scheme = make_tda_covering_scheme( tda_lens_data , cfg_tda_covering_scheme , verbo_lvl )

tda_mapper = km.KeplerMapper( verbose = verbo_lvl )

tda_modelKM = tda_mapper.map( X                      = tda_data ,
                              lens                   = tda_lens_data ,
                              cover                  = tda_covering_scheme ,
                              clusterer              = tda_clusterer ,
                              remove_duplicate_nodes = True )

row_count = tda_data.shape[0]
tda_model_node_data_list , tda_model_edge_data_list = parse_tda_modelKM( tda_modelKM , row_count )

tda_model = dict()
tda_model['data']                = tda_data
tda_model['data_hdr']            = tda_data_hdr
tda_model['lens_data']           = tda_lens_data
tda_model['lens_data_hdr']       = tda_lens_data_hdr
tda_model['cfg_covering_scheme'] = cfg_tda_covering_scheme
tda_model['node_data_list']      = tda_model_node_data_list
tda_model['edge_data_list']      = tda_model_edge_data_list
tda_model['KM']                  = tda_modelKM

filehdl_data_out = open( filename_data_out , 'wb' )
pickle.dump( tda_model ,
             filehdl_data_out ,
             protocol = pickle.HIGHEST_PROTOCOL )
filehdl_data_out.close()

tda_mapper.visualize( tda_modelKM ,
                      path_html = filename_tda_model ,
                      title     = title_tda_model )
