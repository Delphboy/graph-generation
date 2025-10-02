import itertools

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import to_networkx, from_networkx, to_undirected

from generators import generator

class Topology(generator.Generator):
    def __init__(self, args) -> None:
        super().__init__(args)

    def _get_all_possibe_edge_pairings(self, num_nodes): return list(itertools.product(range(num_nodes), repeat=2))
    def _get_all_self_connect_edges(self, num_nodes): return [(x,x) for x in range(num_nodes)]
    def _get_edge_pairings(self, graph): return [(x.item(), y.item()) for x,y in graph.edge_index.T]

    def _get_reverse_connections(self, direct_connections):
        reverse_connections = []
        for v_i, v_j in direct_connections:
            if (v_j, v_i) not in direct_connections:
                reverse_connections.append((v_j, v_i))
        return reverse_connections


    def _build_semantic_graph(self, idx) -> Data:
        data = Data()
        nodes, edges = self._get_vgcap_data(idx)
        objects = self._get_objects(idx)

        assert nodes.shape[0] == objects.shape[0], f"SHAPE MISMATCH: Unable to link nodes ({nodes.shape}) and objects ({objects.shape}) in {idx}. "

        edge_index = edges[:, :2]
        edge_attr = edges[:, 2]

        bad_indicies = []
        for i, (x,y) in enumerate(edge_index):
            if x == y:
                bad_indicies.append(i)

        for bad in  bad_indicies:
            edge_index = np.delete(edge_index, bad, axis=0) 
            edge_attr = np.delete(edge_attr, bad) 

        data.x = objects
        data.edge_index = torch.from_numpy(edge_index.T)
        data.edge_attr = torch.from_numpy(edge_attr)

        return data

    def _build_geometric_graph(self, idx) -> Data:
        data = Data()
        data.x = self._get_objects(idx)
        data.edge_index = self._get_geometric_edges(idx)
        data.edge_attr = self._get_geometric_relationships(idx)
        data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr)
        return data


    def _get_shortest_distance(self, graph: Data, v_i: int, v_j: int) -> int: 
        try:
            shortest_path = nx.shortest_path(to_networkx(graph), v_i, v_j)
            assert len(shortest_path) > 1, "BAD SHORTEST PATH"
        except:
            shortest_path = []
        return max(len(shortest_path) - 1, 0) # Produces number of hops between v_i and v_j

    def _calculate_m(self, graph):
        # max{min dis(v_i, v_j)}

        # TODO: Is m calculated on all possible edge pairings or only the existing ones within graph?
        pairings = self._get_all_possibe_edge_pairings(graph.x.shape[0])
        # pairings = self._get_edge_pairings(graph)
        m = -1
        for v_i, v_j in pairings:
            # if v_i == v_j: continue
            dist = self._get_shortest_distance(graph, v_i, v_j)
            m = dist if dist > m else m
        return m

    def _build_hybrid_graph(self, semantic, geometric):
        """
        we define ⊕ as the following operations: 
        (1) when two objects have both the semantic topology and the geometric topology, we just keep the semantic direction of the connection, e.g., woman → board; 
        (2) we add undirected edges when the semantic graph e^{se}_{ij} has no edges connected, but the geometric graph e^{ge}_{ij} has edges connected, e.g., board − pole

        Attributes are added to the hybrid edges later on, based on the D_{ij} definition in Eq7
        """
        assert semantic.x.shape[0] == geometric.x.shape[0], f"EXPECTED EQUAL NODES: Number of nodes in semantic graph ({semantic.X.shape[0]}) is different to geometric ({geometric.X.shape[0]})"

        hybrid_graph = Data()
        hybrid_graph.x = semantic.x
        hybrid_edges = []

        # Get the set of semantic edges
        sem_edges = [(i.item(), j.item()) for i, j in semantic.edge_index.T]
        inv_sem_edges = [(j.item(), i.item()) for i, j in semantic.edge_index.T]

        # Get the set of geometric edges
        geo_edges = [(i.item(), j.item()) for i, j in geometric.edge_index.T]
        # G - S - inv(S)
        remaining_edges = list(set(geo_edges) - set(sem_edges) - set(inv_sem_edges))

        assert len(remaining_edges) % 2 == 0, f"Remaining edges should be even as it must contain reverse connections to be undirected"

        hybrid_edges += sem_edges
        hybrid_edges += remaining_edges

        hybrid_graph.edge_index = torch.tensor(hybrid_edges).T
        return hybrid_graph

    def build_graph(self, idx: int):
        semantic_graph = self._build_semantic_graph(idx)
        geometric_graph = self._build_geometric_graph(idx)
        hybrid_graph = self._build_hybrid_graph(semantic_graph, geometric_graph)

        m = self._calculate_m(hybrid_graph)

        # Get all possible edge pairings
        all_edge_pairings = self._get_all_possibe_edge_pairings(hybrid_graph.x.shape[0])

        self_connections = self._get_all_self_connect_edges(hybrid_graph.x.shape[0])

        semantic_edge_pairings = self._get_edge_pairings(semantic_graph)
        geometric_edge_pairings = self._get_edge_pairings(geometric_graph)
        direct_connections = list(set(semantic_edge_pairings).union(geometric_edge_pairings))
        reverse_connections = self._get_reverse_connections(direct_connections)

        not_connected = set(all_edge_pairings) - set(self_connections) - set(direct_connections) - set(reverse_connections)
        assert len(not_connected) > 0
        
        # 1. v_i = v_j --> weight = 0
        # 2. directly connected --> weight = min(dis(v_i, v_j))
        # 3. reverse connected --> weight = min(dis(v_i, v_j)) + m
        # 4. not connected --> 2m+1

        topology_edges = []
        topology_weights = []
        for v_i, v_j in self_connections: # Self connections
            topology_edges.append([v_i, v_j])
            topology_weights.append(0)

        for v_i, v_j in direct_connections:
            assert v_i != v_j, f"BAD DIRECT CONNECTION: v_i == v_j ({v_i} == {v_j})"
            topology_edges.append([v_i, v_j])
            topology_weights.append(self._get_shortest_distance(hybrid_graph, v_i, v_j))

        for v_i, v_j in reverse_connections:
            assert v_i != v_j, "BAD REVERSE CONNECTION: v_i == v_j"
            topology_edges.append([v_i, v_j])
            topology_weights.append(self._get_shortest_distance(hybrid_graph, v_i, v_j) + m)

        for v_i, v_j in not_connected:
            topology_edges.append([v_i, v_j])
            topology_weights.append(2*m+1)

        assert len(topology_edges) == len(topology_weights), f"LENGTH MISMATCH: Expected topology_edges ({len(topology_edges)}) to match topology_weights ({len(topology_weights)})"


        edges = torch.tensor(topology_edges).T
        edge_attr = torch.tensor(topology_weights)

        data = HeteroData()
        objects = self._get_objects(idx)
        data["object"].x = objects
        data["object", "topology_relationships", "object"].edge_index = edges
        data["object", "topology_relationships", "object"].edge_attr = edge_attr
        # print(data)
        # print()
        return data

