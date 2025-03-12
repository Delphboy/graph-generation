from torch_geometric.data import HeteroData

from generators import generator

class HeteroYao(generator.Generator):
    def __init__(self, args) -> None:
        super().__init__(args)

    def build_graph(self, idx: int):
        data = HeteroData()
        data["object"].x = self._get_objects(idx)
        data["object", "semantic_relationships", "object"].edge_index = self._get_semantic_edges(idx)
        data["object", "semantic_relationships", "object"].edge_attr = self._get_semantic_relationships(idx)
        data["object", "geometric_relationships", "object"].edge_index = self._get_spatial_edges(idx)
        data["object", "geometric_relationships", "object"].edge_attr = self._get_spatial_relationships(idx)
        # data["attribute"].x = self._get_attribute_nodes(idx)
        return data

