"""Perform GNN model training with uncertainty quantification."""

from copy import deepcopy
from itertools import product
import sys, os, time, argparse
sys.path.insert(0, "../src")

import torch
import toml
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader
seed_everything(42)
from torch import load

from gamenet_uq.training import scale_target, train_loop, test_loop, nll_loss, nll_loss_warmup
from gamenet_uq.nets import GameNetUQ_ablation
from gamenet_uq.post_training import create_model_report
from gamenet_uq.dataset import AdsorptionGraphDataset

# nondeterministic locks
# import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# torch.backends.cudnn.deterministic = True 
# torch.use_deterministic_algorithms(True, warn_only=True)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Perform a training process with the provided hyperparameter settings.")
    PARSER.add_argument("-i", "--input", type=str, dest="i", 
                        help="Input toml file with hyperparameters for the learning process.")
    PARSER.add_argument("-nruns", "--nruns", type=int, dest="nruns", 
                        help="Number of runs for each combinations of features.")
    PARSER.add_argument("-o", "--output", type=str, dest="o", 
                        help="Output directory for the results.")
    ARGS = PARSER.parse_args()

    GCN_OPTIONS = [True, False]
    TS_OPTIONS = [True, False]
    SURF_2HOP = [True, False]    
    feature_combinations = list(product(TS_OPTIONS, GCN_OPTIONS, SURF_2HOP))

    hyperparameters = toml.load(ARGS.i)  
    ase_database_path = hyperparameters["data"]["ase_database_path"]
    graph_dataset_dir = hyperparameters["data"]["graph_dataset_path"]
    graph_settings = hyperparameters["graph"]
    train = hyperparameters["train"]
    architecture = hyperparameters["architecture"]        
    # Select device
    device_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("Device name: {}".format(torch.cuda.get_device_name(0)))
        device_dict["name"] = torch.cuda.get_device_name(0)
        device_dict["CudaDNN_enabled"] = torch.backends.cudnn.enabled
        device_dict["CUDNN_version"] = torch.backends.cudnn.version()
        device_dict["CUDA_version"] = torch.version.cuda
    else:
        print("Device name: CPU")
        device_dict["name"] = "CPU"     

    # Load graph dataset only to get OHE
    dataset = AdsorptionGraphDataset(ase_database_path,
                                    graph_dataset_dir,
                                    graph_settings, 
                                    '')
    ohe_elements = dataset.ohe_elements

    train_loader_ttt = load("../trainings/DATALAODERS/train_loader.pth")
    val_loader_ttt = load("../trainings/DATALAODERS/val_loader.pth")
    test_loader_ttt = load("../trainings/DATALAODERS/test_loader.pth")
    train_datalist_ttt = train_loader_ttt.dataset
    val_datalist_ttt = val_loader_ttt.dataset
    test_datalist_ttt = test_loader_ttt.dataset

    # No 2-hop metal neighbours
    train_loader_ttf = load("../trainings/DATALAODERS/train_loader_ttf.pth")
    val_loader_ttf = load("../trainings/DATALAODERS/val_loader_ttf.pth")
    test_loader_ttf = load("../trainings/DATALAODERS/test_loader_ttf.pth")
    train_datalist_ttf = train_loader_ttf.dataset
    val_datalist_ttf = val_loader_ttf.dataset
    test_datalist_ttf = test_loader_ttf.dataset
    
    for i in range(len(train_datalist_ttt)):
        assert train_datalist_ttt[i].formula == train_datalist_ttf[i].formula

    for i, (TS, GCN, SURF) in enumerate(feature_combinations):
        for j in range(ARGS.nruns):
            torch.manual_seed(42)
            
            if os.path.exists(os.path.join(ARGS.o, "TS_{}_GCN_{}_SURF2HOPS_{}_{}".format(TS, GCN, SURF, j+1))):
                print("TS={}, GCN={}, SURF={}, RUN={} already exists. Skip.".format(TS, GCN, SURF, j+1))
                continue

            print("Run {} of {} for TS={}, GCN={}, SURF={}".format(j+1, ARGS.nruns, TS, GCN, SURF))
            if SURF:
                train_datalist = deepcopy(train_datalist_ttt)
                val_datalist = deepcopy(val_datalist_ttt)
                test_datalist = deepcopy(test_datalist_ttt)
            else:
                train_datalist = deepcopy(train_datalist_ttf)
                val_datalist = deepcopy(val_datalist_ttf)
                test_datalist = deepcopy(test_datalist_ttf)

            # if num_workers > 0, nondeterministic behavior occurs!
            train_loader = DataLoader(train_datalist, batch_size=train["batch_size"], shuffle=True)
            val_loader = DataLoader(val_datalist, batch_size=train["batch_size"], shuffle=False)
            test_loader = DataLoader(test_datalist, batch_size=train["batch_size"], shuffle=False)            
            
            NODE_DIM = 20 if GCN else 19
            
            train_loader, val_loader, test_loader, mean, std = scale_target(train_loader,
                                                                            val_loader,
                                                                            test_loader, 
                                                                            mode=train["target_scaling"], 
                                                                            test=train["test_set"])    
            
            model = GameNetUQ_ablation(node_features=NODE_DIM,              
                            dim=architecture["dim"],
                            num_linear=architecture["num_linear"], 
                            num_conv=architecture["num_conv"],    
                            bias=architecture["bias"], 
                            ts_layer=TS, 
                            gcn=GCN, 
                            seed=42).to(device)
            initial_params = {name: p.clone() for name, p in model.named_parameters()}

            optimizer = torch.optim.Adam(model.parameters(),
                                        lr=train["lr0"],
                                        eps=train["eps"], 
                                        weight_decay=train["weight_decay"],
                                        amsgrad=train["amsgrad"])
            
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                    mode='min',
                                                                    factor=train["factor"],
                                                                    patience=train["patience"],
                                                                    min_lr=train["minlr"])  
            
            loss_list, train_list, val_list, test_list, lr_list = [], [], [], [], []
            train_std, val_std, test_std = [], [], [] 
            try:
                t0 = time.time()
                for epoch in range(1, train["epochs"]+1):
                    lr = lr_scheduler.optimizer.param_groups[0]['lr']        
                    loss_func = nll_loss if epoch > 0 else nll_loss_warmup
                    loss, train_MAE, train_scale = train_loop(model, device, train_loader, optimizer, loss_func)  
                    val_MAE, val_scale = test_loop(model, val_loader, device, std)  
                    lr_scheduler.step(val_MAE)
                    if train["test_set"]:
                        test_MAE, test_scale = test_loop(model, test_loader, device, std)         
                        test_list.append(test_MAE)
                        test_std.append(test_scale)
                        print('Epoch {:03d}: LR={:.7f}  Train MAE: {:.4f} eV  Val MAE: {:.4f} eV '             
                            'Test MAE: {:.4f} eV'.format(epoch, lr, train_MAE*std, val_MAE, test_MAE))
                    else:
                        print('Epoch {:03d}: LR={:.7f}  Train MAE: {:.6f} eV  Val MAE: {:.6f} eV '
                            .format(epoch, lr, train_MAE*std, val_MAE))         
                    loss_list.append(loss)
                    train_list.append(train_MAE * std)
                    train_std.append(train_scale * std)
                    val_list.append(val_MAE)
                    val_std.append(val_scale)
                    lr_list.append(lr)                    
                parameter_changes = {}
                for name, p in model.named_parameters():
                    change = torch.norm(p - initial_params[name])  # L2 norm
                    parameter_changes[name] = change
                sorted_changes = sorted(parameter_changes.items(), key=lambda x: x[1], reverse=True)
                print("-----------------------------------------------------------------------------------------")
                training_time = (time.time() - t0) / 60.0  
                print("Training time: {:.2f} min".format(training_time))
                device_dict["training_time"] = training_time
                create_model_report("TS_{}_GCN_{}_SURF2HOPS_{}_{}".format(TS, GCN, SURF, j+1),
                                    ARGS.o,
                                    hyperparameters,  
                                    model, 
                                    (train_loader, val_loader, test_loader),
                                    (mean, std),
                                    (train_list, val_list, test_list, lr_list),
                                    ohe_elements, 
                                    device_dict, 
                                    parameter_changes, 
                                    (train_std, val_std, test_std))
            except:
                print("Error in training. Break training and go to the next run.")
                break
