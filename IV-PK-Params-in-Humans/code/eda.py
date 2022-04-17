import numpy as np
import matplotlib.pyplot as plt
from util_local import load_data , plotloglog_lenses_pair
from util_global import distance_Thompson_base10 , calc_for_all_pairs_A

filename_data_in = '../data/IV-PK-Params-in-Humans.csv'

_ , data , feature = load_data(filename_data_in)

tda_data = np.nan_to_num( data[feature['2']].to_numpy() , nan = 0.0 )

tda_lens_choice1_descr = feature['2a']
tda_lens_choice1 = data[tda_lens_choice1_descr].to_numpy()

tda_lens_choice2_descr = feature['5']
tda_lens_choice2 = data[tda_lens_choice2_descr].to_numpy()

tda_metric = lambda x1 , x2 : distance_Thompson_base10( x1 , x2 , 'Linf' )
distce = calc_for_all_pairs_A( tda_data , tda_metric )

x = tda_lens_choice1[:,0]
y = tda_lens_choice1[:,1]
x_label = tda_lens_choice1_descr[0]
y_label = tda_lens_choice1_descr[1]
fig_titlebase = 'Lenses pair choice 1'
fig_filename = '../EDA/lenses_pair_1.png'
plotloglog_lenses_pair( x , y , x_label , y_label , fig_titlebase , fig_filename )

x = tda_lens_choice2[:,0]
y = tda_lens_choice2[:,1]
x_label = tda_lens_choice2_descr[0]
y_label = tda_lens_choice2_descr[1]
fig_titlebase = 'Lenses pair choice 2'
fig_filename = '../EDA/lenses_pair_2.png'
plotloglog_lenses_pair( x , y , x_label , y_label , fig_titlebase , fig_filename )

fig , ax = plt.subplots()
plt.hist( distce , bins = 50 , density = True , facecolor = 'g' )
plt.xlabel( 'Pairwise distances' )
plt.ylabel( 'Density' )
plt.savefig( '../EDA/distances_distribution.png' )
plt.show()
plt.close()
