import numpy as np
import torch
import torch_geometric as pyg

from data.ofa_data import OFAPygDataset


class InstagramMentionDataset(OFAPygDataset):
    """
    Node classification on the Instagram mention graph.
    Uses pre-computed 393-dim features (9 account stats + 384 MiniLM embeddings).
    """

    GRAPH_PATH = "/home1/eibl/gfm/prodigy/data/graphs/ukr_ru/instagram/mention_graph_overperformer_minilm.pt"

    def __init__(self, name, load_texts=False, encoder=None, root="./cache_data",
                 transform=None, pre_transform=None):
        super().__init__(name, load_texts=True, encoder=encoder, root=root,
                         transform=transform, pre_transform=pre_transform)

    def gen_data(self):
        ckpt = torch.load(self.GRAPH_PATH, map_location="cpu")
        data_obj = ckpt["data"]

        x = data_obj.x.numpy().astype(np.float32)   # [N, 393]
        y = data_obj.y                                # [N] long, -1 = unlabeled
        edge_index = data_obj.edge_index              # [2, E]
        D = x.shape[1]

        label_names = list(data_obj.label_names)
        num_classes = len(label_names)

        data = pyg.data.Data(edge_index=edge_index, y=y, num_nodes=x.shape[0])
        data.label_names = label_names

        # Class node features: mean feature vector of labeled nodes per class
        class_feats = np.zeros((num_classes, D), dtype=np.float32)
        for c in range(num_classes):
            mask = (y == c).numpy()
            if mask.sum() > 0:
                class_feats[c] = x[mask].mean(axis=0)

        edge_feat    = np.zeros((1, D), dtype=np.float32)
        noi_feat     = np.zeros((1, D), dtype=np.float32)
        prompt_feat  = np.zeros((3, D), dtype=np.float32)

        task_map = {
            "e2e_node": {
                "noi_node_text_feat":    ["noi_node_text_feat",   [0]],
                "class_node_text_feat":  ["class_node_text_feat", torch.arange(num_classes)],
                "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]],
            }
        }

        texts = [x, edge_feat, noi_feat, class_feats, prompt_feat]
        return [data], texts, task_map

    def add_raw_texts(self, data_list, texts):
        data_list[0].node_text_feat        = np.array(texts[0])
        data_list[0].edge_text_feat        = np.array(texts[1])
        data_list[0].noi_node_text_feat    = np.array(texts[2])
        data_list[0].class_node_text_feat  = np.array(texts[3])
        data_list[0].prompt_edge_text_feat = np.array(texts[4])
        return self.collate(data_list)

    def add_text_emb(self, data_list, texts_emb):
        return self.add_raw_texts(data_list, texts_emb)

    def get_task_map(self):
        return self.side_data

    def get_edge_list(self, mode="e2e_node"):
        return {"f2n": [1, [0]], "n2f": [3, [0]], "n2c": [2, [0]], "c2n": [4, [0]]}


class InstagramMentionLanguageDataset(InstagramMentionDataset):
    GRAPH_PATH = "/home1/eibl/gfm/prodigy/data/graphs/ukr_ru/instagram/mention_graph_language_minilm.pt"


class TwitterRetweetRepdemDataset(InstagramMentionDataset):
    """
    Node classification on the Twitter retweet graph with rep/dem pseudo labels.
    Same feature format as Instagram (393-dim: 9 stats + 384 MiniLM).
    """
    GRAPH_PATH = "/home1/eibl/gfm/prodigy/data/graphs/midterm/retweet_graph_repdem_minilm.pt"
