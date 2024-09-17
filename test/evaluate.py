# %%
import os
import sys
sys.path
sys.path.append('../../L1DeepMETv2/')
import time
import os.path as osp
import numpy as np
import json
import torch
import shutil
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.net as net
import model.data_loader as data_loader
from graphmetnetwork import GraphMetNetwork

# %%
n_features_cont = 6
n_features_cat = 2
scale_momentum = 128

# %%
def evaluate(model, loss_fn, dataloader, metrics, model_dir, n_features_cont = 6, save_METarr = True, removePuppi = False):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # summary for current eval loop
    loss_avg_arr = []
    qT_arr = []
    
    MET_arr = {
        'genMETx': [],
        'genMETy': [],
        
        'METx': [],
        'METy': [],
        
        'puppiMETx': [],
        'puppiMETy': []
    }
    
    resolutions_arr = {
        'MET':      [[],[],[]],
        'puppiMET': [[],[],[]],
    }

    colors = {
    #    'pfMET': 'black',
        'puppiMET': 'red',
    #    'deepMETResponse': 'blue',
    #    'deepMETResolution': 'green',
        'MET':  'magenta',
    }

    labels = {
    #    'pfMET': 'PF MET',
        'puppiMET': 'PUPPI MET',
    #    'deepMETResponse': 'DeepMETResponse',
    #    'deepMETResolution': 'DeepMETResolution',
        'MET': 'DeepMETv2'
    }

    weights_pdgId_arr = {
        'down': [],
        'up': [],
        'electron': [],
        'muon': [],
        'photon': [],
        'kaon': [],
        'pion': [],
    }
    
    puppi_weights_pdgId_arr = {
        'down': [],
        'up': [],
        'electron': [],
        'muon': [],
        'photon': [],
        'kaon': [],
        'pion': [],
    }
    
    # compute metrics over the dataset
    for data in tqdm(dataloader, desc="Testing Progress", leave=False):
        
        if removePuppi:
            x_cont = data.x[:,:(n_features_cont-1)]
        else:
            x_cont = data.x[:,:n_features_cont]
        
        x_cat = data.x[:,n_features_cont:].long()
        
        # Convert inputs to numpy arrays
        c_x_cont = np.ascontiguousarray(x_cont.squeeze(0).numpy())
        c_x_cat = np.ascontiguousarray(x_cat.squeeze(0).numpy())
        c_batch = np.ascontiguousarray(data.batch.squeeze(0).numpy())
        num_nodes = x_cont.shape[0]
        
        # Run forward method
        model.GraphMetNetworkLayers(c_x_cont, c_x_cat, c_batch, num_nodes)
        
        # Get result
        result = torch.from_numpy(model.get_output())

        loss = loss_fn(result, data.x, data.y, data.batch)

        # compute all metrics on this batch
        resolutions, METs, weights_pdgId, puppi_weights_pdgId = metrics['resolution'](result, data.x, data.y, data.batch, scale_momentum)
        
        for key in resolutions_arr.keys():
            for i in range(len(resolutions_arr[key])):
                resolutions_arr[key][i]=np.concatenate((resolutions_arr[key][i],resolutions[key][i]))

        for key in MET_arr.keys():
            MET_arr[key]=np.concatenate((MET_arr[key],METs[key]))
            
        for pdg in weights_pdgId_arr.keys():
            weights_pdgId_arr[pdg] = np.concatenate((weights_pdgId_arr[pdg],weights_pdgId[pdg]))
            puppi_weights_pdgId_arr[pdg] = np.concatenate((puppi_weights_pdgId_arr[pdg],puppi_weights_pdgId[pdg]))
            
        qT_arr = np.concatenate((qT_arr, METs['genMET']))
        
        loss_avg_arr.append(loss.item())
    
    print('Done Testing, saving results...')
    if save_METarr:
        for key in MET_arr.keys():
            np.savetxt(f'{model_dir}/epoch_{key}.txt', MET_arr[key].ravel(), delimiter = ',')
        for pdg in weights_pdgId_arr.keys():
            np.savetxt(f'{model_dir}/epoch_{pdg}_weights.txt', weights_pdgId_arr[pdg].ravel(), delimiter = ',')
    
    # compute mean of all metrics in summary
    max_x=400 # max qT value
    x_n=20 # number of bins

    bin_edges=np.arange(0, max_x, max_x/x_n)
    
    inds=np.digitize(qT_arr, bin_edges)

    qT_hist=[]
    for i in range(1, len(bin_edges)):
        qT_hist.append((bin_edges[i]+bin_edges[i-1])/2.)
    
    resolution_hists={}
    for key in resolutions_arr:

        R_arr=resolutions_arr[key][2] 
        u_perp_arr=resolutions_arr[key][0]
        u_par_arr=resolutions_arr[key][1]

        u_perp_hist=[]
        u_perp_scaled_hist=[]
        u_par_hist=[]
        u_par_scaled_hist=[]
        R_hist=[]

        for i in range(1, len(bin_edges)):
            R_i=abs(R_arr[np.where(inds==i)[0]])
            R_hist.append(np.mean(R_i))
            
            u_perp_i=u_perp_arr[np.where(inds==i)[0]]
            u_perp_scaled_i=u_perp_i/np.mean(R_i)
            u_perp_hist.append((np.quantile(u_perp_i,0.84)-np.quantile(u_perp_i,0.16))/2.)
            u_perp_scaled_hist.append((np.quantile(u_perp_scaled_i,0.84)-np.quantile(u_perp_scaled_i,0.16))/2.)
            
            u_par_i=u_par_arr[np.where(inds==i)[0]]
            u_par_scaled_i=u_par_i/np.mean(R_i)
            u_par_hist.append((np.quantile(u_par_i,0.84)-np.quantile(u_par_i,0.16))/2.)
            u_par_scaled_hist.append((np.quantile(u_par_scaled_i,0.84)-np.quantile(u_par_scaled_i,0.16))/2.)

        u_perp_resolution=np.histogram(qT_hist, bins=x_n, range=(0,max_x), weights=u_perp_hist)
        u_perp_scaled_resolution=np.histogram(qT_hist, bins=x_n, range=(0,max_x), weights=u_perp_scaled_hist)
        u_par_resolution=np.histogram(qT_hist, bins=x_n, range=(0,max_x), weights=u_par_hist)
        u_par_scaled_resolution=np.histogram(qT_hist, bins=x_n, range=(0,max_x), weights=u_par_scaled_hist)
        R=np.histogram(qT_hist, bins=x_n, range=(0,max_x), weights=R_hist)
        resolution_hists[key] = {
            'u_perp_resolution': u_perp_resolution,
            'u_perp_scaled_resolution': u_perp_scaled_resolution,
            'u_par_resolution': u_par_resolution,
            'u_par_scaled_resolution':u_par_scaled_resolution,
            'R': R
        }
    
    metrics_mean = {
        'loss': np.mean(loss_avg_arr),
        #'resolution': (np.quantile(resolutions_arr,0.84)-np.quantile(resolutions_arr,0.16))/2.
    }
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    print("- Eval metrics : " + metrics_string)
    
    return metrics_mean, resolution_hists, MET_arr

def main():
    data_dir = '../../L1DeepMETv2/data_ttbar'
    output_dir = "weights_files/"
    dataloaders = data_loader.fetch_dataloader(data_dir = data_dir, batch_size=1, validation_split=.2)
    test_dl = dataloaders['test']
    loss_fn = net.loss_fn
    metrics = net.metrics
    model_dir = 'ckpts_relu'
    
    prefix = '../../L1DeepMETv2/ckpts_April30_scale_sigmoid'
    restore_ckpt = osp.join(prefix, 'best.pth.tar')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    norm = torch.tensor([1., 1., 1., 1., 1., 1.]).to(device=device)
    torch_model = net.Net(continuous_dim=6, categorical_dim=2 , norm=norm).to(device)
    torch_model.eval()
    
    param_restored_new = utils.load_checkpoint(restore_ckpt, torch_model)
    weights_dict = param_restored_new['state_dict']
    print(weights_dict)
    
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

    # %%
    # Create an instance of the C++ GraphMetNetwork model
    model = GraphMetNetwork()

    # Load the weights
    model.load_weights(output_dir)

    # %%
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, model_dir)
    metrics_mean = test_metrics[0]  
    resolutions = test_metrics[1]

    # %%
    # Save metrics in a json file in the model directory
    utils.save_dict_to_json(metrics_mean, osp.join(model_dir, 'metrics_val_best.json'))
    utils.save(resolutions, osp.join(model_dir, 'best.resolutions'))

if __name__ == "__main__":
    main()