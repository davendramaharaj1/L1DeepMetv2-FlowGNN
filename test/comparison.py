import os
import shutil
import sys
import os.path as osp
sys.path
sys.path.append('../../L1DeepMETv2/')
from graphmetnetwork import GraphMetNetwork

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_undirected
from torch_cluster import radius_graph, knn_graph
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from tqdm import tqdm
import model.net as net
import model.data_loader as data_loader
import utils

data_dir = '../../L1DeepMETv2/data_ttbar'
dataloaders = data_loader.fetch_dataloader(data_dir = data_dir, batch_size=1, validation_split=.2)
test_dl = dataloaders['test']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Test dataloader: {}'.format(len(test_dl)))
print(device)

# Load Torch Model
prefix = '../../L1DeepMETv2/ckpts_April30_scale_sigmoid'
restore_ckpt = osp.join(prefix, 'last.pth.tar')
norm = torch.tensor([1., 1., 1., 1., 1., 1.]).to(device=device)
torch_model = net.Net(continuous_dim=6, categorical_dim=2 , norm=norm).to(device)
print(torch_model)

# Get the weights
param_restored_new = utils.load_checkpoint(restore_ckpt, torch_model)
weights_dict = param_restored_new['state_dict']
epoch = param_restored_new['epoch']
torch_model.eval()  # Set the torch model to eval mode
print(weights_dict)

# Store weights in binaries for C model
output_dir = "weights_files/"

# Check if the directory exists
if os.path.exists(output_dir):
    # Iterate over all the files in the directory
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            # Check if it's a file and delete it
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # If it's a directory, delete the directory and its contents
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
else:
    print(f"Directory {output_dir} does not exist.")


# Function to save the weights as binary files
def save_weights_as_binary(weights_dict, output_dir):
    for key, tensor in weights_dict.items():
        # Convert the tensor to a NumPy array
        np_array = tensor.cpu().numpy()

        # Create a binary file name based on the tensor name
        file_name = output_dir + key.replace('.', '_') + '.bin'

        # Save the NumPy array as a binary file
        np_array.tofile(file_name)
        
# Save all weights in the OrderedDict to binary files
save_weights_as_binary(weights_dict, output_dir)

# Create an instance of the C++ GraphMetNetwork model
cmodel = GraphMetNetwork()

# Load the weights
cmodel.load_weights(output_dir)

# Verify weights are the same between torch model and C model
num_weights = 0
for key, tensor in weights_dict.items():
    # Convert the tensor to a NumPy array
    np_array = tensor.cpu().numpy()

    # Return cmodel function pointer to get the weight array
    cmodel_weight_func_name = 'get_' + key.replace('.', '_')
    cmodel_weight_func = getattr(cmodel, cmodel_weight_func_name)
    cmodel_weight_array = cmodel_weight_func()
    
    # Compare Torch model weight with Cmodel weight
    assert(np.allclose(np_array, cmodel_weight_array, atol=1e-5)), f'cmodel.{cmodel_weight_func_name} returned the wrong weights'
    num_weights += 1

print(f'Number of weights checked: {num_weights}')

# Run Inference and compare outputs
import json

# List to track errors
failed_cases = []

# Run the model with input data
counter = 0
for data in tqdm(test_dl, desc="Testing Progress", leave=False):
    data = data.to(device)
    
    x_cont = data.x[:,:6]  # include puppi
    x_cat = data.x[:,6:].long()
    num_nodes = x_cont.shape[0]
    
    etaphi = torch.cat([data.x[:,3][:,None], data.x[:,4][:,None]], dim=1)
    edge_index = radius_graph(etaphi, r=0.4, batch=data.batch, loop=False, max_num_neighbors=255)  # turn off self-loop
    
    x_cont_c = np.ascontiguousarray(x_cont.cpu().numpy())
    x_cat_c = np.ascontiguousarray(x_cat.cpu().numpy())
    batch_c = np.ascontiguousarray(data.batch.cpu().numpy())

    # Run the PyTorch model
    torch_output = torch_model(x_cont, x_cat, edge_index, data.batch)
    torch_output_np = torch_output.detach().cpu().numpy().astype(np.float32)

    # Run the C++ model
    cmodel.GraphMetNetworkLayers(x_cont_c, x_cat_c, batch_c, num_nodes)
    c_output = np.array(cmodel.get_output()).astype(np.float32)

    try:
        # Compare the outputs
        np.testing.assert_allclose(c_output, torch_output_np, rtol=1e-3, err_msg=f'test_dl counter = {counter} failed')
    except AssertionError as e:
        # Log the failed case
        max_abs_diff = np.max(np.abs(c_output - torch_output_np))
        max_rel_diff = np.max(np.abs(c_output - torch_output_np) / np.abs(torch_output_np))
        
        failed_cases.append({
            'counter': counter,
            'c_output': c_output.tolist(),
            'torch_output': torch_output_np.tolist(),
            'max_abs_diff': float(max_abs_diff),
            'max_rel_diff': float(max_rel_diff),
            'error': str(e)
        })
    
    counter += 1

# Save failed cases to a JSON file
with open("failed_cases_relu.json", "w") as f:
    json.dump(failed_cases, f, indent=4)

print(f"Total failed cases: {len(failed_cases)}")