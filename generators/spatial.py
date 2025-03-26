import numpy as np
import torch
from torchvision.ops.boxes import box_iou
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse

from generators import generator

class Sptial(generator.Generator):
    def __init__(self, args) -> None:
        super().__init__(args)

    def _generate_spatial_adjacency_matrix(self, bboxes):
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

        return adjacency_matrix


    def build_graph(self, idx: int):
        data = HeteroData()

        bbox_data = self._get_bboxes(idx)

        adjacency_matrix = self._generate_spatial_adjacency_matrix(bbox_data)
        adjacency_matrix = torch.from_numpy(adjacency_matrix)

        edge_index, edge_attr = dense_to_sparse(adjacency_matrix)

        data["object"].x = self._get_objects(idx)
        data["object", "spatial_relationships", "object"].edge_index = edge_index
        data["object", "spatial_relationships", "object"].edge_attr = edge_attr

        return data

