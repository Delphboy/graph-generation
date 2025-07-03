import os

import multiprocessing
import numpy as np
import torch
from torchvision.ops.boxes import box_iou
from torch_geometric.utils import dense_to_sparse

class Generator():
    def __init__(self, args) -> None:
        self.butd = args.butd
        self.vsua = args.vsua
        self.sgae = args.sgae
        self.vgcap = args.vgcap
        self.bbox = args.bbox if args.bbox != "NOT PROVIDED" else None
        self.output_dir = args.output_dir

    def _get_objects(self, idx:int ):
        return torch.from_numpy(
            np.load(os.path.join(self.butd, str(idx)+".npz"))['feat']
        )

    def _get_bboxes(self, idx:int ):
        assert self.bbox is not None, "MISSING FLAG: Please ensure that the --bbox flag is set"
        return torch.from_numpy(
            np.load(os.path.join(self.bbox, str(idx)+".npy"))
        )

    def _get_geometric_graph_data(self, idx):
        return np.load(os.path.join(self.vsua, str(idx) + '.npy'),
                       allow_pickle=True,
                       encoding='latin1').item()

    def _get_geometric_edges(self, idx):
        return torch.from_numpy(self._get_geometric_graph_data(idx)["edges"]).type(torch.long).t().contiguous()

    def _get_geometric_relationships(self, idx):
        return torch.from_numpy(self._get_geometric_graph_data(idx)["feats"]).type(torch.float32)

    def _get_semantic_graph_data(self, idx:int):
        return np.load(os.path.join(self.sgae, str(idx) + '.npy'),
                       allow_pickle=True,
                       encoding='latin1').item()

    def _get_attribute_nodes(self, idx):
        semantic_graph_data = self._get_semantic_graph_data(idx)
        attribute_nodes = torch.from_numpy(semantic_graph_data["obj_attr"]).type(torch.long)
        return attribute_nodes[:, 1:]

    def _get_semantic_edges(self, idx):
        semantic_graph_data = self._get_semantic_graph_data(idx)
        semantic_edges = torch.from_numpy(semantic_graph_data["rela_matrix"]).type(torch.long)
        return semantic_edges[:, :-1].t().contiguous()

    def _get_semantic_relationships(self, idx):
        semantic_graph_data = self._get_semantic_graph_data(idx)
        semantic_edges = torch.from_numpy(semantic_graph_data["rela_matrix"]).type(torch.long)
        semantic_relationships = semantic_edges[:, -1].reshape([-1, 1]).type(torch.long)
        return semantic_relationships

    def _generate_spatial_edges_and_features(self, bboxes):
        adjacency_matrix = np.zeros((len(bboxes),len(bboxes)))

        for a in range(len(bboxes)):
            for b in range(len(bboxes)):
                if a == b: continue

                bbox_a = bboxes[a]
                bbox_a_x1, bbox_a_y1, bbox_a_x2, bbox_a_y2 = bbox_a

                bbox_b = bboxes[b]
                bbox_b_x1, bbox_b_y1, bbox_b_x2, bbox_b_y2 = bbox_b

                # Check if bbox_a is inside of bbox_b
                if bbox_a_x1 > bbox_b_x1 and bbox_a_y1 > bbox_b_y1 and bbox_a_x2 < bbox_b_x2 and bbox_a_y2 < bbox_b_y2:
                    adjacency_matrix[a][b] = 1

                # Check if bbox_a is outside of bbox_b
                elif bbox_a_x1 < bbox_b_x1 and bbox_a_y1 < bbox_b_y1 and bbox_a_x2 > bbox_b_x2 and bbox_a_y2 > bbox_b_y2:
                    adjacency_matrix[a][b] = 2

                # Check if bbox_a and bbox_b have an IoU of >= 0.5
                elif box_iou(bbox_a.unsqueeze(0), bbox_b.unsqueeze(0)) >= 0.5:
                    adjacency_matrix[a][b] = 3

                else:
                    centroid_a = torch.tensor([bbox_a_x1 + abs(bbox_a_x1 - bbox_a_x2) / 2, bbox_a_y1 + abs(bbox_a_y1 - bbox_a_x2) / 2])
                    centroid_b = torch.tensor([bbox_b_x1 + abs(bbox_b_x1 - bbox_b_x2) / 2, bbox_b_y1 + abs(bbox_b_y1 - bbox_b_y2) / 2])

                    vecAB = centroid_b - centroid_a
                    hoz = torch.tensor([1, 0], dtype=torch.float)

                    inner = torch.inner(vecAB, hoz)
                    norms = torch.linalg.norm(vecAB) * torch.linalg.norm(hoz)

                    cos = inner / norms
                    rad = torch.acos(torch.clamp(cos, -1.0, 1.0))
                    deg = torch.rad2deg(rad)

                    adjacency_matrix[a,b] = torch.ceil(deg/45) + 3

        adjacency_matrix = torch.from_numpy(adjacency_matrix)
        edge_index, edge_attr = dense_to_sparse(adjacency_matrix)
        edge_attr = edge_attr.to(torch.long)
        return edge_index, edge_attr
    
    def _get_vgcap_data(self, idx: int):
        data = np.load(os.path.join(self.vgcap, str(idx) + ".npz"))
        nodes = data["obj"]
        edges = np.concatenate([data["prela"], data["wrela"]], axis=0)
        # if the relation of an image is empty, then fill in it with <0, 0, 'near'> to avoid problems
        if edges.shape[0] == 0:
            edges = np.array([[0, 0, 119]], dtype=int) # 119 == near
        return nodes, edges

    def build_graph(self, idx: int):
        raise NotImplementedError(f"The function `build_graph` was called with IDX={idx} but has not been implemented")

    def build_graph_wrapper(self, args):
        idx, output_dir, _ = args
        print(f"Processing: {idx}")
        try:
            graph = self.build_graph(idx)
            torch.save(graph, os.path.join(output_dir, idx)+".pt")
        except Exception as e:
            print(f"EXCEPTION OCCURED WHILST PROCESSING GRAPH={idx}")
            raise e

    def build_graphs(self):
        file_names = os.listdir(self.butd)
        idxs = [file_name.split('.')[0] for file_name in file_names]
        # print(idxs)

        with multiprocessing.Pool(5) as pool:
            args = [(idx, self.output_dir, self.butd) for idx in idxs]
            pool.map(self.build_graph_wrapper, args)

