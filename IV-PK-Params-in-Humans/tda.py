"""
=============
Breast Cancer
=============

This example generates a Mapper built from the `Wisconsin Breast Cancer Dataset`_.

.. _Wisconsin Breast Cancer Dataset: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data


The reasoning behind the choice of lenses in the demonstration below is:

- **For lens1:** Lenses that make biological sense; in other words, lenses that highlight special features in the data, that I know about.
- **For lens2:** Lenses that disperse the data, as opposed to clustering many points together.

In the case of this particular data, using an anomaly score (in this case calculated using the IsolationForest from sklearn) makes biological sense since cancer cells are anomalous. For the second lens, we use the :math:`l^2` norm.

For an interactive exploration of lens for the breast cancer, see the `Choosing a lens notebook <../../notebooks/Cancer-demo.html>`_.

KeplerMapper also permits setting multiple datapoint color functions and node color functions in its html visualizations.
The example code below demonstrates three ways this might be done. The rendered visualizations are also viewable:

- `Visualization of the breat cancer mapper using multiple datapoint color functions <../../_static/breast-cancer-multiple-color-functions.html>`_
- `Visualization of the breat cancer mapper using multiple node color functions <../../_static/breast-cancer-multiple-node-color-functions.html>`_
- `Visualization of the breat cancer mapper using multiple datapoint and node color functions <../../_static/breast-cancer-multiple-color-functions-and-multiple-node-color-functions.html>`_

.. image:: ../../../examples/breast-cancer/breast-cancer.png


"""

import pandas as pd
import numpy as np
import kmapper as km
from util import nan_Thompson_base10_distance
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

tda_data = data[feature2].to_numpy()

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

tda_metric = lambda x , y : nan_Thompson_base10_distance( x , y , "Linf" )

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
