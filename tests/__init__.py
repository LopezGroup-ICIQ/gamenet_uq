import pathlib
    
from ase.io import read

from gamenet_uq.nets import GameNetUQ

TEST_DIR = pathlib.Path(__file__).parent

model_test_path = str(TEST_DIR) +  "/files/gamenetuq_558k_250e_14oct2025"
x1 = read(str(TEST_DIR) + "/files/CONTCAR1", format="vasp")
x2 = read(str(TEST_DIR) + "/files/CONTCAR2", format="vasp")
x3 = read(str(TEST_DIR) + "/files/CONTCAR3", format="vasp")
x4 = read(str(TEST_DIR) + "/files/CONTCAR4", format="vasp")
xx = [x1, x2, x3, x4]
f = GameNetUQ(20, 192)

__all__ = [
    "xx", 
    "f", 
    "model_test_path"
]