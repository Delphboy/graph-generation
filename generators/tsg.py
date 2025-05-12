import torch
from torch_geometric import transforms
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected

from generators import generator

class Tsg(generator.Generator):
    def __init__(self, args) -> None:
        super().__init__(args)

    def build_graph(self, idx: int):
        data = HeteroData()

        # objects
        data["object"].x = self._get_objects(idx)
        num_objects = data["object"].x.shape[0]

        node_indices = torch.arange(num_objects)
        src, dst = torch.meshgrid(node_indices, node_indices, indexing="ij")
        complete_edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
        assert complete_edge_index.shape[1] == num_objects + num_objects * (num_objects - 1), f"TSG - BAD OBJECT-ATTENTION-OBJECT: Expected {num_objects + num_objects * (num_objects - 1)} Received {complete_edge_index.shape[1]}"
        data["object", "attention", "object"].edge_index = complete_edge_index

        # semantic relationships
        sem_edges = self._get_semantic_edges(idx)
        sem_attrs = self._get_semantic_relationships(idx)
        edge_index, edge_attr = to_undirected(sem_edges, sem_attrs)
        data["object", "semantic_relationships", "object"].edge_index = edge_index
        data["object", "semantic_relationships", "object"].edge_attr = edge_attr

        attributes = self._get_attribute_nodes(idx)[:,:3].reshape(-1)
        data["attribute"].x = attributes

        attribute_edges = torch.tensor([[i % num_objects, i] for i in range(3*num_objects)]).T

        assert attribute_edges[0].max() < num_objects, "TSG - ATTRIBUTE EDGES HAVE INDEX OUT OF BOUNDS OBJECT"
        assert attribute_edges[1].max() < len(attributes), f"TSG - ATTRIBUTE EDGES HAVE INDEX OUT OF BOUNDS ATTRIBUTE: Should have {attribute_edges[1].max()} < {len(attributes)}"

        data["object", "has_attribute", "attribute"].edge_index = attribute_edges

        # HACK: To get around to_undirect() not working, introduce reverse edges
        data["attribute", "rev_has_attribute", "object"].edge_index, _ = attribute_edges
        rev_attribute_edges = attribute_edges[[1,0],:]
        assert attribute_edges.shape == rev_attribute_edges.shape, f"TSG - REV_ATTRIBUTES HAVE BAD SHAPE"
        data["attribute", "rev_has_attribute", "object"].edge_index = rev_attribute_edges

        return data

