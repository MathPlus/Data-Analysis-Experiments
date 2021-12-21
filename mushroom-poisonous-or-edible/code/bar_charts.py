import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

def df_catg_stats( df , list_colname ) :
    
    catg_stats = dict()
    
    for colname in list_colname :
        
        value_count = df[colname].value_counts().to_dict()
        value_freq  = df[colname].value_counts(normalize=True).to_dict()
        
        list_uniqval = [*value_count]
        
        catg_stats[colname] = { uniqval : { 'count' : value_count[uniqval] ,
                                            'freq' : value_freq[uniqval] }
                                for uniqval in list_uniqval }
    
    return catg_stats


plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['figure.autolayout'] = True

tick_pct_fmt = mtick.PercentFormatter( xmax     = 1.0 ,
                                       decimals = 0 ,
                                       symbol   = '%' ,
                                       is_latex = False )

dirname_fig = '../figures/features_bar_charts'
filename_data_in = '../data-in/data.csv'

data_in = pd.read_csv(filename_data_in)

list_colname = data_in.columns.to_list()

catg_stats = df_catg_stats( df = data_in , list_colname = list_colname )

label_x = [ 'Number of instances' , 'Frequency' ]
label_y = 'Unique values'

for colname in list_colname :

    label_title  = [ '%s : absolute distribution' % colname ,
                     '%s : relative distribution' % colname ]
    filename_fig = [ '%s/absolute/%s - absolute distribution.png' % ( dirname_fig , colname ) ,
                     '%s/relative/%s - relative distribution.png' % ( dirname_fig , colname ) ]
    
    list_uniqval = [ *catg_stats[colname] ]
    list_count = [ catg_stats[colname][uniqval]['count'] for uniqval in list_uniqval ]
    list_idx_sorted_count = list( np.argsort( list_count ) )
    y = [ list_uniqval[idx] for idx in list_idx_sorted_count ]
    
    x = [ [ catg_stats[colname][uniqval]['count'] for uniqval in y ] ,
          [ catg_stats[colname][uniqval]['freq']  for uniqval in y ] ]
    
    for i in range(2) :
        fig , ax = plt.subplots()
        plt.title( label_title[i] )
        plt.xlabel( label_x[i] )
        plt.ylabel( label_y    )
        if i == 1 : ax.xaxis.set_major_formatter(tick_pct_fmt)
        plt.barh( y , x[i] )
        fig.savefig( filename_fig[i] )
