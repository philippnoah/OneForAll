import os.path as osp

import numpy as np
import torch
import torch_geometric as pyg

from data.ofa_data import OFAPygDataset, pth_safe_load, safe_mkdir

# TODO: update this path once graph_retweet/graph_data.pt is in its final location
GRAPH_PATH = "/home1/eibl/gfm/OneForAll/data/midterm/graph_retweet/graph_data.pt"


class MidtermRetweetDataset(OFAPygDataset):
    """
    Node classification on the midterm Twitter retweet graph.

    Node features: pre-computed 86-dim tabular + text-PCA features (from generate_graph_retweet.py).
    Labels: US state (50 classes, -1 for unlabeled nodes).

    All feature types (node, edge, noi, class, prompt) use the same 86-dim space so no
    LLM encoder is needed. Class node embeddings are per-state mean feature vectors.
    """

    def __init__(self, name, load_texts=False, encoder=None, root="./cache_data",
                 transform=None, pre_transform=None):
        # Force load_texts=True so OFAPygDataset skips the encoder and calls add_raw_texts.
        # The encoder argument is accepted but never used.
        super().__init__(name, load_texts=True, encoder=encoder, root=root,
                         transform=transform, pre_transform=pre_transform)

    # ── data generation ───────────────────────────────────────────────────────

    def gen_data(self):
        raw = torch.load(GRAPH_PATH, map_location="cpu")
        x = raw["x"].numpy().astype(np.float32)   # [N, D]
        y = raw["y"]                                # [N] long, -1 = unlabeled
        edge_index = raw["edge_index"]              # [2, E]
        label_names = raw["label_names"]            # list of 50 state strings
        num_classes = len(label_names)
        D = x.shape[1]

        data = pyg.data.Data(edge_index=edge_index, y=y, num_nodes=x.shape[0])
        data.label_names = label_names

        # Class node embeddings: mean feature vector of labeled nodes per state.
        # Falls back to zeros for states with no labeled nodes.
        class_feats = np.zeros((num_classes, D), dtype=np.float32)
        for c in range(num_classes):
            mask = (y == c).numpy()
            if mask.sum() > 0:
                class_feats[c] = x[mask].mean(axis=0)

        # All non-node feature types are zero vectors of the same dimension.
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

        # texts[i] are passed directly to add_raw_texts (no LLM encoding happens)
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
        # Not used (load_texts is always True), but required by the abstract base class.
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
