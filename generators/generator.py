import os

import multiprocessing
import numpy as np
import torch

class Generator():
    def __init__(self, args) -> None:
        self.butd = args.butd
        self.vsua = args.vsua
        self.sgae = args.sgae
        self.output_dir = args.output_dir

    def _get_objects(self, idx:int ):
        return torch.from_numpy(
            np.load(os.path.join(self.butd, str(idx)+".npz"))['feat']
        )

    def _get_spatial_graph_data(self, idx):
        return np.load(os.path.join(self.vsua, str(idx) + '.npy'),
                       allow_pickle=True,
                       encoding='latin1').item()

    def _get_spatial_edges(self, idx):
        return torch.from_numpy(self._get_spatial_graph_data(idx)["edges"]).type(torch.long).t().contiguous()

    def _get_spatial_relationships(self, idx):
        return torch.from_numpy(self._get_spatial_graph_data(idx)["feats"]).type(torch.float32)

    def _get_semantic_graph_data(self, idx:int):
        return np.load(os.path.join(self.sgae, str(idx) + '.npy'),
                       allow_pickle=True,
                       encoding='latin1').item()

    def _get_attribute_nodes(self, idx):
        semantic_graph_data = self._get_semantic_graph_data(idx)
        attribute_nodes = torch.from_numpy(semantic_graph_data["obj_attr"]).type(torch.float32)
        return torch.unique(attribute_nodes[:, 1:])

    def _get_semantic_edges(self, idx):
        semantic_graph_data = self._get_semantic_graph_data(idx)
        semantic_edges = torch.from_numpy(semantic_graph_data["rela_matrix"]).type(torch.long)
        return semantic_edges[:, :-1].t().contiguous()

    def _get_semantic_relationships(self, idx):
        semantic_graph_data = self._get_semantic_graph_data(idx)
        semantic_edges = torch.from_numpy(semantic_graph_data["rela_matrix"]).type(torch.long)
        semantic_relationships = semantic_edges[:, -1].reshape([-1, 1]).type(torch.float32)
        return semantic_relationships

    def build_graph(self, idx: int):
        raise NotImplementedError(f"The function `build_graph` was called with IDX={idx} but has not been implemented")


    def build_graph_wrapper(self, args):
        idx, output_dir, _ = args
        print(f"Processing: {idx}")
        graph = self.build_graph(idx)
        torch.save(graph, os.path.join(output_dir, idx)+".pt")

    def build_graphs(self):
        file_names = os.listdir(self.butd)
        idxs = [file_name.split('.')[0] for file_name in file_names]

        with multiprocessing.Pool(5) as pool:
            args = [(idx, self.output_dir, self.butd) for idx in idxs]
            pool.map(self.build_graph_wrapper, args)

