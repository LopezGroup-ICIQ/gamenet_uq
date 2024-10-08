"""
Module for post-processing and collecting results from the GNN model training.
"""

import os
from datetime import date, datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
from sklearn.metrics import r2_score

from gamenet_uq.graph_tools import graph_plotter
from gamenet_uq.plot_functions import *


def create_model_report(model_name: str,
                        model_path: str,
                        configuration_dict: dict,
                        model: torch.nn.Module,
                        loaders: tuple[DataLoader],                     
                        scaling_params: tuple[float], 
                        mae_lists: tuple[list], 
                        one_hot_encoder_elements, 
                        device: dict=None, 
                        params_changes: dict=None, 
                        std_lists: tuple[list]=None,
                        save_loaders: bool=True):
    """Create full report of the performed GNN training.

    Args:
        model_name (str): name of the model.
        model_path (str): path to the model folder.
        configuration_dict (dict): input hyperparams dict from toml input file.
        model (_type_): model object.
        loaders (tuple[DataLoader]): train/val/test sets(loaders).
        scaling_params (tuple[float]): Scaling params (mean and std of train+val sets).
        mae_lists (tuple[list]): MAE trends of train/val/test sets during learning process.
        device (dict, optional): Dictionary containing device info. Defaults to None.

    Returns:
        (str): Confirmation that model has been saved.   
    """
    print("Saving the model ...")
    
    # 1) Get time of the run
    today = date.today()
    today_str = str(today.strftime("%d-%b-%Y"))
    time = str(datetime.now())[11:]
    time = time[:8]
    run_period = "{}, {}\n".format(today_str, time)
        
    # 2) Unfold  train/val/test dataloaders
    train_loader, val_loader, test_loader = loaders[0], loaders[1], loaders[2]
    N_train, N_val = len(train_loader.dataset), len(val_loader.dataset)

    # 3) Unfold hyperparameters
    graph = configuration_dict["graph"]
    train = configuration_dict["train"]
    architecture = configuration_dict["architecture"]
    
    # 4) Extract graph conversion parameters
    voronoi_tol = graph["structure"]["tolerance"]
    second_order_nn = graph["structure"]["second_order"]
    scaling_factor = graph["structure"]["scaling_factor"]
    node_adsorbate_descriptor = graph["features"]["adsorbate"]
    node_radical_descriptor = graph["features"]["radical"]
    
    # 5) Extract model scaling parameters
    if train["target_scaling"] == "std":
        mean_tv = scaling_params[0]
        std_tv = scaling_params[1]
    else:
        pass

    if std_lists != None:
        train_std = std_lists[0]
        val_std = std_lists[1]
        test_std = std_lists[2]
    
    # 6) Create directory structure where to store model training results
    try:
        os.mkdir("{}/{}".format(model_path, model_name))
    except FileExistsError:
        model_name = input("The name defined already exists in the provided directory: Provide a new one: ")
        os.mkdir("{}/{}".format(model_path, model_name))
    os.mkdir("{}/{}/Outliers".format(model_path, model_name))

    # 7) Store info of device on which model training has been performed
    if device != None:
        with open('{}/{}/device.txt'.format(model_path, model_name), 'w') as f:
            print(device, file=f)

    # 8) Store params changes
    if params_changes != None:
        with open('{}/{}/params_changes.txt'.format(model_path, model_name), 'w') as f:
            print(params_changes, file=f)

    # 8) Get predictions and true values and save them in a csv file for train/val sets
    model.eval()
    model.to("cpu")
    x_pred, x_true = [], []  # Train set
    a_pred, a_true = [], []  # Validation set
    
    for graph in train_loader.dataset:  # iter graph by graph to avoid reshuffling here
        with torch.no_grad():
            x_pred += model(graph).mean
            x_true += graph.target
    for batch in val_loader:
        batch = batch.to("cpu")
        with torch.no_grad():
            a_pred += model(batch).mean
            a_true += batch.target
    # Re-scale predictions and true values
    z_pred = [x_pred[i].item()*std_tv + mean_tv for i in range(N_train)]  # Train set
    z_true = [x_true[i].item() for i in range(N_train)]
    b_pred = [a_pred[i].item()*std_tv + mean_tv for i in range(N_val)]  # Val set
    b_true = [a_true[i].item() for i in range(N_val)]
    error_train = [(z_pred[i] - z_true[i]) for i in range(N_train)]                   # Error (train set)
    error_val = [(b_pred[i] - b_true[i]) for i in range(N_val)]                       # Error (validation set)
    abs_error_train = [abs(error_train[i]) for i in range(N_train)]                   # Absolute Error (train set)
    abs_error_val = [abs(error_val[i]) for i in range(N_val)]                         # Absolute Error (val set)
    # Get data labels in train/val sets
    train_label_list = [graph.formula for graph in train_loader.dataset]
    val_label_list = [graph.formula for graph in val_loader.dataset]
    train_facet_list = [graph.facet for graph in train_loader.dataset]
    val_facet_list = [graph.facet for graph in val_loader.dataset]
    train_bb_list = [graph.bb_type for graph in train_loader.dataset]
    val_bb_list = [graph.bb_type for graph in val_loader.dataset]
    train_metal_list = [graph.metal for graph in train_loader.dataset]
    val_metal_list = [graph.metal for graph in val_loader.dataset]
    
    with open("{}/{}/train_set.csv".format(model_path, model_name), "w") as file4:
        writer = csv.writer(file4, delimiter='\t')
        writer.writerow(["System", "Metal", "Surface", "Bond", "True_eV", "Prediction_eV", "Error_eV", "Abs_error_eV"])
        writer.writerows(zip(train_label_list, train_metal_list, train_facet_list, train_bb_list, z_true, z_pred, error_train, abs_error_train))    
    with open("{}/{}/validation_set.csv".format(model_path, model_name), "w") as file4:
        writer = csv.writer(file4, delimiter='\t')
        writer.writerow(["System", "Metal", "Surface", "Bond", "True_eV", "Prediction_eV", "Error_eV", "Abs_error_eV"])
        writer.writerows(zip(val_label_list, val_metal_list, val_facet_list, val_bb_list, b_true, b_pred, error_val, abs_error_val))

    # MAE trend during training
    train_list = mae_lists[0]
    val_list = mae_lists[1]
    test_list = mae_lists[2]
    lr_list = mae_lists[3]

    # 9) Save dataloaders for future use
    if save_loaders:
        torch.save(train_loader, "{}/{}/train_loader.pth".format(model_path, model_name))
        torch.save(val_loader, "{}/{}/val_loader.pth".format(model_path, model_name))
    
    # 10) Save model architecture and parameters
    torch.save(model, "{}/{}/model.pth".format(model_path, model_name))             # Save model architecture
    torch.save(model.state_dict(), "{}/{}/GNN.pth".format(model_path, model_name))  # Save model parameters
    #save one hot encoder
    torch.save(one_hot_encoder_elements, "{}/{}/one_hot_encoder_elements.pth".format(model_path, model_name))
        
            
    # 11) Store Hyperparameters dict from input file
    with open('{}/{}/input.txt'.format(model_path, model_name), 'w') as g:
        print(configuration_dict, file=g)


    # 12) Store train_list, val_list, and lr_list as .csv file
    with open('{}/{}/training.csv'.format(model_path, model_name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if train["test_set"] == False:
            writer.writerow(["Epoch", "Train_MAE_eV", "Val_MAE_eV", "Train_std_eV", "Val_std_eV", "Learning_Rate"])
            for i in range(len(train_list)):
                writer.writerow([i+1, train_list[i], val_list[i], train_std[i], val_std[i], lr_list[i]])
        else:
            writer.writerow(["Epoch", "Train_MAE_eV", "Val_MAE_eV", "Test_MAE_eV", "Train_std_eV", "Val_std_eV", "Test_std_eV", "Learning_Rate"])
            for i in range(len(train_list)):
                writer.writerow([i+1, train_list[i], val_list[i], test_list[i], train_std[i], val_std[i], test_std[i], lr_list[i]])

    
    loss = train["loss_function"] 
    if train["test_set"] == False: 
        N_tot = N_train + N_val
        file1 = open("{}/{}/performance.txt".format(model_path, model_name), "w")
        file1.write("GRAPH REPRESENTATION PARAMETERS\n")
        file1.write("Voronoi tolerance = {} Angstrom\n".format(voronoi_tol))
        file1.write("Atomic radii scaling factor = {}\n".format(scaling_factor))
        file1.write("Second order metal neighbours inclusion = {}\n".format(second_order_nn))
        file1.write("Node adsorbate/surface descriptor = {}\n".format(node_adsorbate_descriptor))
        file1.write("Node radical descriptor = {}\n".format(node_radical_descriptor))
        file1.write("TRAINING PROCESS\n")
        file1.write(run_period)
        file1.write("Dataset Size = {}\n".format(N_tot))
        file1.write("Data Split (Train/Val) = {}-{} %\n".format(*split_percentage(train["splits"], train["test_set"])))
        file1.write("Target scaling = {}\n".format(train["target_scaling"]))
        file1.write("Dataset (train+val) mean = {:.6f} eV\n".format(scaling_params[0]))
        file1.write("Dataset (train+val) standard deviation = {:.6f} eV\n".format(scaling_params[1]))
        file1.write("Epochs = {}\n".format(train["epochs"]))
        file1.write("Batch Size = {}\n".format(train["batch_size"]))
        file1.write("Optimizer = Adam\n")                                            # Kept fixed in this project
        file1.write("Learning Rate scheduler = Reduce Loss On Plateau\n")            # Kept fixed in this project
        file1.write("Initial Learning Rate = {}\n".format(train["lr0"]))
        file1.write("Minimum Learning Rate = {}\n".format(train["minlr"]))
        file1.write("Patience (lr-scheduler) = {}\n".format(train["patience"]))
        file1.write("Factor (lr-scheduler) = {}\n".format(train["factor"]))
        file1.write("Loss function = {}\n".format(loss))
        file1.close()
        return "Model saved in {}/{}".format(model_path, model_name)
    
    # 13) Get info from test set if it has been monitored
    if save_loaders:
        torch.save(test_loader, "{}/{}/test_loader.pth".format(model_path, model_name))
    test_label_list = [graph.formula for graph in test_loader.dataset]
    test_facet_list = [graph.facet for graph in test_loader.dataset]
    test_bb_list = [graph.bb_type for graph in test_loader.dataset]
    test_metal_list = [graph.metal for graph in test_loader.dataset]
    N_test = len(test_loader.dataset)  
    N_tot = N_train + N_val + N_test    
    w_pred, w_true = [], []  # Test set
    for batch in test_loader:
        with torch.no_grad():
            batch = batch.to("cpu")
            w_pred += model(batch).mean
            w_true += batch.target
    y_pred = [w_pred[i].item()*std_tv + mean_tv for i in range(N_test)]  # Test set
    y_true = [w_true[i].item() for i in range(N_test)] 
    error_test = [(y_pred[i] - y_true[i]) for i in range(N_test)]                     # Error (test set)
    abs_error_test = [abs(error_test[i]) for i in range(N_test)]                      # Absolute Error (test set)
    squared_error_test = [error_test[i] ** 2 for i in range(N_test)]                  # Squared Error
    abs_pctg_error_test = [abs(error_test[i] / y_true[i]) for i in range(N_test)]     # Absolute Percentage Error
    std_error_test = np.std(error_test)                                               # eV
    # Save test set error of the samples            
    with open("{}/{}/test_set.csv".format(model_path, model_name), "w") as file4:
        writer = csv.writer(file4, delimiter='\t')
        writer.writerow(["System", "Metal", "Surface", "Bond", "True_eV", "Prediction_eV", "Error_eV", "Abs_error_eV"])
        writer.writerows(zip(test_label_list, test_metal_list, test_facet_list, test_bb_list, y_true, y_pred, error_test, abs_error_test))   

    formula, metal, surface, bb, y_true, y_mean, y_std, y_min, y_max, in_interval, error = [], [], [], [], [], [], [], [], [], [], []
    for graph in test_loader.dataset:
        formula.append(graph.formula)
        metal.append(graph.metal)
        surface.append(graph.facet)
        bb.append(graph.bb_type)
        y_true.append(graph.target.numpy()[0])
        y_mean.append(model(graph).mean.cpu().detach().numpy()[0] * std_tv + mean_tv)
        y_std.append(model(graph).stddev.cpu().detach().numpy()[0] * std_tv)
        # Get confidence interval for each prediction based on the std and mean at confidence level 95%
        y_min.append(y_mean[-1] - 1.96 * y_std[-1])
        y_max.append(y_mean[-1] + 1.96 * y_std[-1])
        # Check if the true value is in the confidence interval
        in_interval.append(
            (y_min[-1] < y_true[-1] < y_max[-1])
        )
        error.append(y_true[-1] - y_mean[-1])

    df = pd.DataFrame(
        {
            "formula": formula,
            "metal": metal,
            "surface": surface,
            "bb": bb,
            "y_true": y_true,
            "y_mean": y_mean,
            "error": error,
            "y_std": y_std,
            "y_min": y_min,
            "y_max": y_max,
            "in_interval": in_interval,
        }
    )
    df["norm_res"] = df["error"] / df["y_std"]
    sha = ((np.sum(df["y_std"].pow(2))) / len(df)) ** 0.5  # Sharpness [eV]
    mu_std = df["y_std"].mean()
    cv = (np.sum(df["y_std"].array - mu_std) ** 2.0 / (len(df) - 1.0)) ** 0.5 / mu_std  # Coefficient of variation [-]
    x = np.linspace(-6.0, 6.0, 100000)
    CDF_observed = [np.sum(df["norm_res"] < i) / len(df) for i in x]
    CDF_theoretical = (1 + torch.erf(torch.tensor(x / np.sqrt(2)))) / 2
    CDF_observed = np.array(CDF_observed)
    CDF_theoretical = np.array(CDF_theoretical)
    CDF_observed[np.where(CDF_observed == 0)] = 1e-10
    CDF_observed[np.where(CDF_observed == 1)] = 1 - 1e-10
    CDF_theoretical[np.where(CDF_theoretical == 0)] = 1e-10
    CDF_theoretical[np.where(CDF_theoretical == 1)] = 1 - 1e-10
    miscalibration_area = np.sum(np.abs(CDF_observed - CDF_theoretical)) / len(CDF_observed)
    df.to_csv("{}/{}/uq.csv".format(model_path, model_name), index=False)
    # Performance Report
    file1 = open("{}/{}/performance.txt".format(model_path, model_name), "w")
    file1.write(run_period)
    if device is not None:
        file1.write("Device = {}\n".format(device["name"]))
        file1.write("Training time = {:.2f} min\n".format(device["training_time"]))
    file1.write("---------------------------------------------------------\n")
    file1.write("GRAPH REPRESENTATION PARAMETERS\n")
    file1.write("Voronoi tolerance = {} Angstrom\n".format(voronoi_tol))
    file1.write("Atomic radius scaling factor = {}\n".format(scaling_factor))
    file1.write("Second order metal neighbours inclusion = {}\n".format(second_order_nn))
    file1.write("Node adsorbate/surface descriptor = {}\n".format(node_adsorbate_descriptor))
    file1.write("Node radical descriptor = {}\n".format(node_radical_descriptor))
    file1.write("---------------------------------------------------------\n")
    file1.write("GNN ARCHITECTURE\n")
    file1.write("Activation function = {}\n".format(architecture["sigma"]))
    # file1.write("Convolutional layer = {}\n".format(architecture["conv_layer"]))
    #file1.write("Pooling layer = {}\n".format(architecture["pool_layer"]))
    file1.write("Number of convolutional layers = {}\n".format(architecture["num_conv"]))
    file1.write("Number of fully connected layers = {}\n".format(architecture["num_linear"]))
    file1.write("Depth of the layers = {}\n".format(architecture["dim"]))
    file1.write("Bias presence in the layers = {}\n".format(architecture["bias"]))
    file1.write("---------------------------------------------------------\n")
    file1.write("TRAINING PROCESS\n")
    file1.write("Dataset Size = {}\n".format(N_tot))
    file1.write("Data Split (Train/Val/Test) = {}-{}-{} %\n".format(*split_percentage(train["splits"])))
    file1.write("Target scaling = {}\n".format(train["target_scaling"]))
    file1.write("Target (train+val) mean = {:.6f} eV\n".format(mean_tv))
    file1.write("Target (train+val) standard deviation = {:.6f} eV\n".format(std_tv))
    file1.write("Epochs = {}\n".format(train["epochs"]))
    file1.write("Batch size = {}\n".format(train["batch_size"]))
    file1.write("Optimizer = Adam\n")                                            # Kept fixed in this project
    file1.write("Learning Rate scheduler = Reduce Loss On Plateau\n")            # Kept fixed in this project
    file1.write("Initial learning rate = {}\n".format(train["lr0"]))
    file1.write("Minimum learning rate = {}\n".format(train["minlr"]))
    file1.write("Patience (lr-scheduler) = {}\n".format(train["patience"]))
    file1.write("Factor (lr-scheduler) = {}\n".format(train["factor"]))
    file1.write("Loss function = {}\n".format(loss))
    file1.write("---------------------------------------------------------\n")
    file1.write("GNN PERFORMANCE\n")
    file1.write("Test set size = {}\n".format(N_test))
    file1.write("Mean Bias Error (MBE) = {:.3f} eV\n".format(np.mean(error_test)))
    file1.write("Mean Absolute Error (MAE) = {:.3f} eV\n".format(np.mean(abs_error_test)))
    file1.write("Root Mean Square Error (RMSE) = {:.3f} eV\n".format(np.sqrt(np.mean(squared_error_test))))
    file1.write("Mean Absolute Percentage Error (MAPE) = {:.3f} %\n".format(np.mean(abs_pctg_error_test)*100.0))
    file1.write("Error Standard Deviation = {:.3f} eV\n".format(np.std(error_test)))
    file1.write("R2 = {:.3f} \n".format(r2_score(y_true, y_pred)))
    file1.write("Sharpness = {:.3f} eV\n".format(sha))
    file1.write("Coefficient of variation = {:.3f} [-]\n".format(cv))
    file1.write("Miscalibration area = {:.3f} [-]\n".format(miscalibration_area))
    file1.write("---------------------------------------------------------\n")
    file1.write("OUTLIERS (TEST SET)\n")
    # 15) Get outliers
    outliers_list, outliers_error_list, index_list = [], [], []
    counter = 0
    for sample in range(N_test):
        if abs_error_test[sample] >= 3 * std_error_test:  
            counter += 1
            outliers_list.append(test_label_list[sample])
            outliers_error_list.append(error_test[sample])
            index_list.append(sample)
            if counter < 10:
                file1.write("0{}) {}    Error: {:.2f} eV    (index={})\n".format(counter, test_label_list[sample], error_test[sample], sample))
            else:
                file1.write("{}) {}    Error: {:.2f} eV    (index={})\n".format(counter, test_label_list[sample], error_test[sample], sample))
            text = "{}\nError: {:.2f} eV".format(test_label_list[sample], error_test[sample])
            graph_plotter(test_loader.dataset[sample], text=text, node_index=False)
            plt.savefig("{}/{}/Outliers/{}.svg".format(model_path, model_name, test_label_list[sample].strip()))
            plt.close()
    file1.close()    
    return "Model saved in {}/{}".format(model_path, model_name)