import torch
from torch_geometric.data import HeteroData

from generators import generator

class HeteroVsua(generator.Generator):
    def __init__(self, args) -> None:
        super().__init__(args)

    def build_graph(self, idx: int):
        data = HeteroData()

        data["object"].x = self._get_objects(idx)
        num_objects = data["object"].x.shape[0]

        data["object", "semantic_relationships", "object"].edge_index = self._get_semantic_edges(idx)
        data["object", "semantic_relationships", "object"].edge_attr = self._get_semantic_relationships(idx)

        data["object", "geometric_relationships", "object"].edge_index = self._get_geometric_edges(idx)
        data["object", "geometric_relationships", "object"].edge_attr = self._get_geometric_relationships(idx)

        attributes = self._get_attribute_nodes(idx)[:,:3].reshape(-1)
        data["attribute"].x = attributes

        attribute_edges = torch.tensor([[i % num_objects, i] for i in range(3*num_objects)]).T
        data["object", "has_attribute", "attribute"].edge_index = attribute_edges

        return data

