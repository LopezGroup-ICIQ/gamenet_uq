## Download the DFT datasets

DFT data used to train and evaluate GAME-Net UQ are publicly available in Zenodo as ASE databases:

- `fg.db`: Contains the original FG-dataset extended with additional surfaces and open-shell fragments
- `ts.db`: Transition state dataset evaluated with the improved dimer method, for bond-breaking/forming event with up to C2O3 intermediates on the most stable surface of 13 metals (all contained in the FG-dataset except Fe)
- `int.db`: The same intermediates defining the initial states of the TS evaluated in the `ts.db`.

## Analyze DFT datasets

TODO

## Generate graph datasets

Once the ASE databases are downloaded, you can directly convert the ASE Atoms objects to `torch_geometric.Data` instances by running the script `gen_dataset.py`.
You have to provide the following conversion settings:

- `tol`:
- `sf`:

TODO

## Analyze graph datasets

TODO

