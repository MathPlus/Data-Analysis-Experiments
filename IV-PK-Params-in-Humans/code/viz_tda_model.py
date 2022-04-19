from pyvis.network import Network as PyVisNet
import networkx as nx


def viz_tda_model(tda_model) :
    
    nx_graph = nx.Graph()
    
    nx__node_data_list = list(enumerate( tda_model['node_data_list'] ))
    nx__edge_data_list = [ ( edge_data['nodeA_idx'] , edge_data['nodeB_idx'] , edge_data ) for edge_data in tda_model['edge_data_list'] ]
    
    nx_graph.add_nodes_from( nx__node_data_list )
    nx_graph.add_edges_from( nx__edge_data_list )
    
    nx.draw(nx_graph)
    
    pv_graph = PyVisNet('600px', '600px')
    pv_graph.from_nx(nx_graph)
    pv_graph.show('test.html')
    
    # nx_graph.nodes[1]['title'] = 'Number 1'
    # nx_graph.nodes[1]['group'] = 1
    # nx_graph.nodes[3]['title'] = 'I belong to a different group!'
    # nx_graph.nodes[3]['group'] = 10
    # nx_graph.add_node(20, size=20, title='couple', group=2)
    # nx_graph.add_node(21, size=15, title='couple', group=2)
    # nx_graph.add_edge(20, 21, weight=5)
    # nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
    # nx.draw(nx_graph)
    # pv_graph = PyVisNet('500px', '500px')
    # pv_graph.from_nx(nx_graph)
    # pv_graph.show('pyvis_example.html')
    return None


import pickle
filename_tda_model = '../TDA_models/IV-PK-Params-in-Humans.pckl'
filehdl_tda_model = open( filename_tda_model , 'rb' )
tda_model = pickle.load(filehdl_tda_model)
filehdl_tda_model.close()


viz_tda_model(tda_model)
