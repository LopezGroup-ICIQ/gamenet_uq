""" Test generated dataset with pytest"""
import sys
import unittest
sys.path.append("../src")
import argparse

from torch import load, Tensor
from torch_geometric.data import Data, InMemoryDataset

from gamenet_uq.dataset import AdsorptionGraphDataset

ASE_DB_PATH = "../data/fg.db"
GRAPH_DATASET_PATH = "../data"
STRUCTURE_DICT = {"tolerance": 0.25, "scaling_factor": 1.25, "second_order": True}
FEATURES_DICT = {"adsorbate": False, "radical": False, "valence": False, "gcn": True, "magnetization": False}
GRAPH_PARAMS = {"structure": STRUCTURE_DICT, "features": FEATURES_DICT, "target": "scaled_energy"}
DB_KEY = ''

dataset = AdsorptionGraphDataset(ASE_DB_PATH, GRAPH_DATASET_PATH, GRAPH_PARAMS, DB_KEY)

class TestDataset(unittest.TestCase):
    def test_undirected(self):
        """
        Check that all the graphs are undirected
        """
        for graph in dataset:
            self.assertTrue(graph.is_undirected())
		
    def test_type(self):
        """
        Check that all the graphs are of type torch_geometric.data.Data
        """
        self.assertIsInstance(dataset, InMemoryDataset)
        for graph in dataset:
            self.assertIsInstance(graph, Data)

    def test_graphs(self):
        """
        Check that the dataset contains the expected number of graphs
        """
        for graph in dataset:
            self.assertTrue(len(graph.elem) != 0)
            self.assertTrue(graph.target == graph.y)
            self.assertTrue(-700.0 < graph.y < 20.0)


class TestNodes(unittest.TestCase):
    def test_node_features(self):
        """
        Check that the node features are of the expected type
        """
        for graph in dataset:
            self.assertIsInstance(graph.x, Tensor)
            self.assertIsInstance(graph.y, Tensor)
            self.assertIsInstance(graph.edge_index, Tensor)

            # custom node attrs
            self.assertIsInstance(graph.formula, str)
            self.assertIsInstance(graph.metal, str)
            self.assertIsInstance(graph.facet, str)
            self.assertIsInstance(graph.elem, list)
            self.assertIsInstance(graph.type, str)
            self.assertIsInstance(graph.bb_type, str)
            self.assertIsInstance(graph.img_freqs, str)
            self.assertIsInstance(graph.e_mol, Tensor)

            # check generalized coordination number
            for node in graph.x:
                gcn_idx = graph.node_feats.index('gcn')
                self.assertTrue(0.0 <= node[gcn_idx].item() <= 1.0)

class TestEdges(unittest.TestCase):
    def test_edge_features(self):
        """
        Check that the edge features are of the expected type
        """
        for graph in dataset:
            self.assertIsInstance(graph.edge_attr, Tensor)
            self.assertTrue(graph.edge_attr.shape[1] == 1)  # maybe define edge as OHE
            self.assertTrue(graph.edge_attr.shape[0] == graph.edge_index.shape[1])
            if graph.type == 'ts':
                # assert that in edge_attr the number of ones is exactly 2
                self.assertTrue(graph.edge_attr.sum() == 2)
