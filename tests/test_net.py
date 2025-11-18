import unittest

import numpy as np
import torch
from torch_geometric.data import Data

from tests import xx, model_test_path
from gamenet_uq.graph import atoms_to_data, get_voronoi_neighbourlist
from gamenet_uq.functions import load_model_from_path, load_model_from_url
from gamenet_uq.node_featurizers import get_gcn
from gamenet_uq.nets import GameNetUQ
from gamenet_uq.constants import ADSORBATE_ELEMS


class TestGraph(unittest.TestCase):
    def test_voronoi(self):
        """
        Test neighbour-list function.
        """
        for x in xx:
            nl = get_voronoi_neighbourlist(x, 0.25, 1.25, ADSORBATE_ELEMS)
            self.assertIsInstance(nl, np.ndarray)
            self.assertEqual(nl.shape[1], 2)

    def test_gcn(self):
        """
        Test implementation of generalized coordination number (gcn).
        """
        for x in xx:
            gcn = get_gcn(x)
            self.assertEqual(len(gcn), len(x), f"Length mismatch! Expected {len(x)}, got {len(gcn)}")
            adsorbate = np.array([atom.symbol in ADSORBATE_ELEMS for atom in x])
            slab = np.array([atom.symbol not in ADSORBATE_ELEMS for atom in x])
            adsorbate_values = gcn[adsorbate]
            slab_values = gcn[slab]
            assert np.all(adsorbate_values == 0), \
            f"Non-zero GCN found for adsorbate atoms: {adsorbate_values[adsorbate_values != 0]}"
            assert np.all(slab_values != 0), \
            f"Zero GCN found for slab atoms: {slab_values[slab_values == 0]}"
            assert np.all((gcn >= 0) & (gcn <= 1)), f"Values found outside [0, 1] in: {gcn}"

    def test_graph_conversion(self):
        for x in xx:
            g = atoms_to_data(x, surface_order=2)
            g_all = atoms_to_data(x, surface_order=-1)
            g_surf_hops = atoms_to_data(x, surface_order=-1, add_surf_hops_info=True)
            self.assertIsInstance(g, Data)
            self.assertTrue("formula" in g.keys())
            self.assertTrue("elem" in g.keys())
            self.assertTrue("ase_indices" in g.keys())
            self.assertTrue("type" in g.keys())
            self.assertTrue("surf_hops" not in g.keys())
            self.assertTrue(len(g.ase_indices) == g.num_nodes)
            self.assertEqual(g.x.shape[1], 20)
            self.assertEqual(g_all.x.shape[1], 20)
            self.assertTrue(g.x.shape[0] <= len(x))
            self.assertTrue(g_all.x.shape[0] == len(x))
            self.assertEqual(g.edge_attr.shape[1], 1)
            self.assertEqual(g_all.edge_attr.shape[1], 1)
            self.assertTrue("surf_hops" in g_surf_hops.keys())
            self.assertIsInstance(g_surf_hops.surf_hops, dict)
            self.assertIsInstance(g_surf_hops.surf_hops[0], list)
            self.assertIsInstance(g_surf_hops.surf_hops[1], list)

class TestNet(unittest.TestCase):
    def test_net(self):
        x = atoms_to_data(xx[1])
        f = GameNetUQ(20, 192, return_distribution=True)
        self.assertIsInstance(f(x), torch.distributions.Normal)
        f = GameNetUQ(20, 192, return_distribution=False)
        y = f(x)
        self.assertIsInstance(y, tuple)
        self.assertTrue(y[1] >= 0.0)

class TestLoad(unittest.TestCase):
    def test_load_from_path(self):
        f = load_model_from_path(model_test_path)
        self.assertIsInstance(f, GameNetUQ)
        self.assertTrue("mean" in f.y_scale_params.keys())
        self.assertTrue("std" in f.y_scale_params.keys())
        self.assertIsInstance(f.y_scale_params["mean"], float)
        self.assertIsInstance(f.y_scale_params["std"], float)
        self.assertGreaterEqual(f.y_scale_params["std"], 0.0)
        x = atoms_to_data(xx[1])
        self.assertIsInstance(f(x), tuple)

    def test_load_from_url(self):
        f = load_model_from_url()
        self.assertIsInstance(f, GameNetUQ)
        self.assertTrue("mean" in f.y_scale_params.keys())
        self.assertTrue("std" in f.y_scale_params.keys())
        self.assertIsInstance(f.y_scale_params["mean"], float)
        self.assertIsInstance(f.y_scale_params["std"], float)
        self.assertGreaterEqual(f.y_scale_params["std"], 0.0)
        x = atoms_to_data(xx[1])
        self.assertIsInstance(f(x), tuple)
