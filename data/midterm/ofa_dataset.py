import os.path as osp

import numpy as np
import torch
import torch_geometric as pyg

from data.ofa_data import OFAPygDataset, pth_safe_load, safe_mkdir


class MidtermRetweetDataset(OFAPygDataset):
    """
    Node classification on the midterm Twitter retweet graph.

    Node features: pre-computed 98-dim tabular + text-PCA features.
    Subclasses set GRAPH_PATH to point to the desired graph_data.pt file.
    """

    GRAPH_PATH = "/home1/eibl/gfm/OneForAll/data/midterm/graph_retweet/graph_data.pt"

    def __init__(self, name, load_texts=False, encoder=None, root="./cache_data",
                 transform=None, pre_transform=None):
        super().__init__(name, load_texts=True, encoder=encoder, root=root,
                         transform=transform, pre_transform=pre_transform)

    # ── data generation ───────────────────────────────────────────────────────

    def gen_data(self):
        raw = torch.load(self.GRAPH_PATH, map_location="cpu")
        x = raw["x"].numpy().astype(np.float32)   # [N, D]
        y = raw["y"]                                # [N] long, -1 = unlabeled
        edge_index = raw["edge_index"]              # [2, E]
        D = x.shape[1]

        # Normalise label_names to a list indexed by class index (0..C-1),
        # handling both list and dict formats.
        raw_label_names = raw["label_names"]
        if isinstance(raw_label_names, dict):
            num_classes = max(k for k in raw_label_names if k >= 0) + 1
            label_names = [raw_label_names.get(i, str(i)) for i in range(num_classes)]
        else:
            label_names = list(raw_label_names)
            num_classes = len(label_names)

        data = pyg.data.Data(edge_index=edge_index, y=y, num_nodes=x.shape[0])
        data.label_names = label_names

        # Class node embeddings: mean feature vector of labeled nodes per class.
        class_feats = np.zeros((num_classes, D), dtype=np.float32)
        for c in range(num_classes):
            mask = (y == c).numpy()
            if mask.sum() > 0:
                class_feats[c] = x[mask].mean(axis=0)

        edge_feat = np.zeros((1, D), dtype=np.float32)
        noi_feat = np.zeros((1, D), dtype=np.float32)
        prompt_feat = np.zeros((3, D), dtype=np.float32)

        task_map = {
            "e2e_node": {
                "noi_node_text_feat": ["noi_node_text_feat", [0]],
                "class_node_text_feat": ["class_node_text_feat", torch.arange(num_classes)],
                "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]],
            }
        }

        texts = [x, edge_feat, noi_feat, class_feats, prompt_feat]
        return [data], texts, task_map

    # ── feature assignment ────────────────────────────────────────────────────

    def add_raw_texts(self, data_list, texts):
        data_list[0].node_text_feat = np.array(texts[0])
        data_list[0].edge_text_feat = np.array(texts[1])
        data_list[0].noi_node_text_feat = np.array(texts[2])
        data_list[0].class_node_text_feat = np.array(texts[3])
        data_list[0].prompt_edge_text_feat = np.array(texts[4])
        return self.collate(data_list)

    def add_text_emb(self, data_list, texts_emb):
        return self.add_raw_texts(data_list, texts_emb)

    # ── task interface ────────────────────────────────────────────────────────

    def get_task_map(self):
        return self.side_data

    def get_edge_list(self, mode="e2e_node"):
        if mode == "e2e_node":
            return {"f2n": [1, [0]], "n2f": [3, [0]], "n2c": [2, [0]], "c2n": [4, [0]]}
        if mode == "lr_node":
            return {"f2n": [1, [0]], "n2f": [3, [0]]}
        return {"f2n": [1, [0]], "n2f": [3, [0]], "n2c": [2, [0]], "c2n": [4, [0]]}


class MidtermRetweetPseudoDataset(MidtermRetweetDataset):
    """Same graph, hashtag-based political orientation pseudo labels (rep=0, dem=1)."""

    GRAPH_PATH = "/home1/eibl/gfm/OneForAll/data/midterm/graph_retweet/graph_data_pseudo.pt"
