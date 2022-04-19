# https://pyvis.readthedocs.io/en/latest/tutorial.html#networkx-integration

from pyvis.network import Network as PyVisNet
import networkx as nx

nx_graph = nx.cycle_graph(10)
nx_graph.nodes[1]['title'] = 'Number 1'
nx_graph.nodes[1]['group'] = 1
nx_graph.nodes[3]['title'] = 'I belong to a different group!'
nx_graph.nodes[3]['group'] = 10
nx_graph.add_node(20, size=20, title='couple', group=2)
nx_graph.add_node(21, size=15, title='couple', group=2)
nx_graph.add_edge(20, 21, weight=5)
nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
nx.draw(nx_graph)
pv_graph = PyVisNet('500px', '500px')
pv_graph.from_nx(nx_graph)
pv_graph.show('pyvis_example.html')