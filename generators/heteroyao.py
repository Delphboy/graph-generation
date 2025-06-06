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

        bbox_data = self._get_bboxes(idx)
        edge_index, edge_attr = self._generate_spatial_edges_and_features(bbox_data)
        data["object", "spatial_relationships", "object"].edge_index = edge_index
        data["object", "spatial_relationships", "object"].edge_attr = edge_attr

        return data

