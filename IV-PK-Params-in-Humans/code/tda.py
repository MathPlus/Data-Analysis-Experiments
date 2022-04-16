"""
========================================================================================================
Trend Analysis of a Database of Intravenous Pharmacokinetic Parameters in Humans for 1352 Drug Compounds
========================================================================================================
"""


import numpy as np
import kmapper as km
from util_global import weblink_fmt , distance_Thompson_base10
from util_local import load_data , make_tda_covering_scheme
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

tda_data = np.nan_to_num( data[feature['2']].to_numpy() , nan = 0.0 )

tda_metric = lambda x , y : distance_Thompson_base10( x , y , 'Linf' )

tda_clusterer = sklearn.cluster.DBSCAN( eps           = 0.5 ,
                                        min_samples   = 5 ,
                                        metric        = tda_metric ,
                                        metric_params = None ,
                                        algorithm     = 'auto' ,
                                        leaf_size     = 30 ,
                                        p             = None ,
                                        n_jobs        = None )

tda_lens = np.log2( data[feature['5']].to_numpy() )
# min(tda_lens[:,0]) == -3.77595972578207
# max(tda_lens[:,0]) == 18.194602975157967
# min(tda_lens[:,1]) == -4.832890014164741
# max(tda_lens[:,1]) == 15.189531985610547
precfg_tda_covering_scheme = dict()
precfg_tda_covering_scheme['lens_bound_rounding0'] = 0.5
precfg_tda_covering_scheme['lens_bound_rounding1'] = 0.5
precfg_tda_covering_scheme['intvls_count0'] = 8
precfg_tda_covering_scheme['intvls_count1'] = 8
precfg_tda_covering_scheme['intvls_overlap0'] = 0.4
precfg_tda_covering_scheme['intvls_overlap1'] = 0.4

tda_covering_scheme = make_tda_covering_scheme( tda_lens , precfg_tda_covering_scheme , verbo_lvl )

tda_mapper = km.KeplerMapper( verbose = verbo_lvl )

tda_model = tda_mapper.map( X                      = tda_data ,
                            lens                   = tda_lens ,
                            cover                  = tda_covering_scheme ,
                            clusterer              = tda_clusterer ,
                            remove_duplicate_nodes = True )

filehdl_data_out = open( filename_data_out , 'wb' )
pickle.dump( tda_model , filehdl_data_out , protocol = pickle.HIGHEST_PROTOCOL )
filehdl_data_out.close()

tda_mapper.visualize( tda_model ,
                      path_html = filename_tda_model ,
                      title     = title_tda_model )
