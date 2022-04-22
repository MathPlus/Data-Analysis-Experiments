from pyvis.network import Network as PyVisNet
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale
from util_global import rgba_0_1_to_0_255


def viz_tda_model(tda_model) :
    
    cm = plt.cm.jet
    
    node_data_list = tda_model['node_data_list']

    node_size_list = [ node_data['node_size/dataset_size'] for node_data in node_data_list ]
    cm_idx = np.round( 255 * minmax_scale( np.array(node_size_list) , feature_range = (0,1) ) ).astype(int)
    
    nx_graph = nx.Graph()
    
    nx__node_data_list = list(enumerate(node_data_list))
    nx__edge_data_list = [ ( edge_data['nodeA_idx'] , edge_data['nodeB_idx'] , edge_data )
                           for edge_data in tda_model['edge_data_list'] ]
    
    nx_graph.add_nodes_from( nx__node_data_list )
    nx_graph.add_edges_from( nx__edge_data_list )
    
    for node_idx , node_data in nx__node_data_list :
        node_size = max( 5.0 , 50.0 * node_data['node_size/dataset_size'] )
        node_title = 'node {node_idx}'.format( node_idx = node_idx )
        node_label = ' '
        node_color_rgba_0_255 = rgba_0_1_to_0_255(cm(cm_idx[node_idx]))
        node_color = 'rgba' + str(node_color_rgba_0_255)
        nx_graph.add_node( node_idx , size  = node_size ,
                                      title = node_title ,
                                      label = node_label ,
                                      color = node_color )
    
    for edge_data in tda_model['edge_data_list'] :
        edge_title = 'edge ({nodeA_idx},{nodeB_idx})'.format( nodeA_idx = edge_data['nodeA_idx'] ,
                                                              nodeB_idx = edge_data['nodeB_idx'] )
        edge_width = max( 1.0 , 8.0 * edge_data['edge_size/nodesAB_size'] )
        nx_graph.add_edge( edge_data['nodeA_idx'] , edge_data['nodeB_idx'] ,
                           title = edge_title ,
                           width = edge_width ,
                           color = 'grey')
    
    # nx.draw(nx_graph)
    
    pv_graph = PyVisNet('600px', '600px')
    pv_graph.from_nx(nx_graph)
    pv_graph.toggle_physics(True)
    pv_graph_html = pv_graph.generate_html().replace( ', "label": " "' , '' ) \
                                            .replace( ', "label": 1'   , '' )
    with open( 'test.html' , 'wt' ) as filehdl_tda_model_html:
        filehdl_tda_model_html.write( pv_graph_html )
    
    return pv_graph , nx_graph


import pickle
filename_tda_model = '../TDA_models/IV-PK-Params-in-Humans.pckl'
filehdl_tda_model = open( filename_tda_model , 'rb' )
tda_model = pickle.load(filehdl_tda_model)
filehdl_tda_model.close()


pv_graph , nx_graph = viz_tda_model(tda_model)
