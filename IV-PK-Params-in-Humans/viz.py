
from kmapper.visuals import colorscale_from_matplotlib_cmap
import matplotlib.pyplot as plt

from my_kmapper_viz import my_kmapper_viz

filename_tda_model = "TDA/IV-PK-Params-in-Humans.html"
title_tda_model    = 'A topological model of the dataset studied in <a href="https://dmd.aspetjournals.org/content/46/11/1466" target="_blank"><i>Trend Analysis of a Database of Intravenous Pharmacokinetic Parameters in Humans for 1352 Drug Compounds</i></a>'

tda_colorscale = colorscale_from_matplotlib_cmap(plt.cm.jet, nbins=255)

def viztda( tda_mapper , tda_model , filename_tda_model , title_tda_model , tda_color_data , tda_color_descr ) :
    tda_mapper.visualize( tda_model ,
                          path_html           = filename_tda_model ,
                          title               = title_tda_model ,
                          color_values        = tda_color_data ,
                          color_function_name = tda_color_descr ,
                          node_color_function = [ "nanmean" , "nanmedian" ] ,
                          colorscale          = tda_colorscale ,
                          nbins               = 20 ,
                          include_searchbar   = True )

def myviztda( tda_mapper , tda_model , filename_tda_model , title_tda_model , tda_color_data , tda_color_descr ) :
    my_kmapper_viz( tda_model ,
                    path_html           = filename_tda_model ,
                    title               = title_tda_model ,
                    color_values        = tda_color_data ,
                    color_function_name = tda_color_descr ,
                    node_color_function = [ "nanmean" , "nanmedian" ] ,
                    colorscale          = tda_colorscale ,
                    nbins               = 20 ,
                    include_searchbar   = True ,
                    verbose             = 1 )

