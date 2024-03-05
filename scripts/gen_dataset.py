import sys
sys.path.append('../src')

from gamenet_uq.dataset import AdsorptionGraphDataset

ASE_DB_PATH = "../data/fg.db"
GRAPH_DATASET_PATH = "../data"
STRUCTURE_DICT = {"tolerance": 0.25, "scaling_factor": 1.25, "second_order": True}
FEATURES_DICT = {"adsorbate": False, "radical": False, "valence": False, "gcn": True, "magnetization": False}
GRAPH_PARAMS = {"structure": STRUCTURE_DICT, "features": FEATURES_DICT, "target": "scaled_energy"}
DB_KEY = ''

if __name__ == "__main__":
    dataset = AdsorptionGraphDataset(ASE_DB_PATH, GRAPH_DATASET_PATH, GRAPH_PARAMS, DB_KEY)
