"""

========================================================================================================
Trend Analysis of a Database of Intravenous Pharmacokinetic Parameters in Humans for 1352 Drug Compounds
========================================================================================================

1352 molecules in total
5 attributes for TDA
    * "human VDss (L/kg)"
    * "human CL (mL/min/kg)"
    * "fraction unbound in plasma (fu)"
    * "MRT (h)"
    * "terminal  t1/2 (h)"
TDA lenses: We use attributes
    * "human VDss (L/kg)"
    * "human CL (mL/min/kg)"
39/1352 (2.88%) molecules have missingness within the two attributes that serve as TDA lenses.
    * 37 molecules have missingness in "human VDss (L/kg)"
    * 2 molecules have missingness in "human CL (mL/min/kg)"
They are discarded.
1313 molecules remain.

426/1313 (32.44%) molecules have missingness within the fives attributes used for TDA.
    * No molecules have missingness in "human VDss (L/kg)" and "human CL (mL/min/kg)" due to the above preprocessing.
    * 413 molecules have missingness in "fraction unbound in plasma (fu)"
    * 4 molecules have missingness in "MRT (h)"
    * 14 molecules have missingness in "terminal  t1/2 (h)"
All non-missing values are positive, ranging from 0.0002 to 1400.0.
Missing values are replaced with 0.0.
This is not imputation. It is an accomodation of the clusterer as it rejects data with missingness.
It does not impact the analysis for the following reason.
    * The clusterer uses our implementation of the Thompson metric.
    * This implementation is tolerant of missing values, and nonpositive values are treated as missing values.

"""

import pandas as pd
import numpy as np
import kmapper as km
from util import distance_Thompson_base10
import sklearn

filename_data_in   = "Data-in/IV-PK-Params-in-Humans.csv"
filename_tda_model = "TDA/IV-PK-Params-in-Humans.html"
title_tda_model    = "Trend Analysis of a Database of Intravenous Pharmacokinetic Parameters in Humans for 1352 Drug Compounds"

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

tda_data = np.nan_to_num( data[feature2].to_numpy() )

tda_lens = np.log2( data[feature2a].to_numpy() )

cfg_tda_covering_scheme = dict()

# min(tda_lens[:,0]) == -5.058893689053568
# max(tda_lens[:,0]) ==  9.451211111832329
# min(tda_lens[:,1]) == -8.0782590139205
# max(tda_lens[:,1]) == 10.06339508128851

tda_intvls_lowerbound0 = -5.5
tda_intvls_upperbound0 =  9.5
tda_intvls_lowerbound1 = -8.5
tda_intvls_upperbound1 = 10.5

tda_intvls_count0 = 12
tda_intvls_count1 = 15

tda_intvls_overlap0 = 0.4
tda_intvls_overlap1 = 0.4

cfg_tda_covering_scheme['bound'] = \
    np.array( [ [ tda_intvls_lowerbound0 , tda_intvls_upperbound0 ] ,
                [ tda_intvls_lowerbound1 , tda_intvls_upperbound1 ] ] )

cfg_tda_covering_scheme['count'] = \
    [ tda_intvls_count0 , tda_intvls_count1 ]

cfg_tda_covering_scheme['overlap'] = \
    [ tda_intvls_overlap0 , tda_intvls_overlap1 ]

tda_covering_scheme = km.Cover( limits       = cfg_tda_covering_scheme['bound'] ,
                                n_cubes      = cfg_tda_covering_scheme['count'] ,
                                perc_overlap = cfg_tda_covering_scheme['overlap'] ,
                                verbose      = 2 )

tda_metric = lambda x , y : distance_Thompson_base10( x , y , "Linf" )

tda_clusterer = sklearn.cluster.DBSCAN( eps           = 0.1 ,
                                        min_samples   = 5 ,
                                        metric        = tda_metric ,
                                        metric_params = None ,
                                        algorithm     = 'auto' ,
                                        leaf_size     = 30 ,
                                        p             = None ,
                                        n_jobs        = None )

tda_mapper = km.KeplerMapper(verbose=2)

tda_model = tda_mapper.map( X                      = tda_data ,
                            lens                   = tda_lens ,
                            cover                  = tda_covering_scheme ,
                            clusterer              = tda_clusterer ,
                            remove_duplicate_nodes = True )

tda_mapper.visualize( tda_model ,
                      path_html = filename_tda_model ,
                      title     = title_tda_model )
