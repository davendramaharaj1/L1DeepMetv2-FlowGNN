import sys
sys.path
sys.path.append('../../L1DeepMETv2/')
import time
import os.path as osp
import torch
from tqdm import tqdm
import utils
from torch_cluster import radius_graph
import model.net as net
import model.data_loader as data_loader

def measure_inference_time(model, dataloader, device='cpu', warmup_batches=10):
    """
    Measure the inference time on either CPU or GPU, with optional GPU warmup.
    
    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (torch.utils.data.DataLoader): Dataloader with the test data.
        device (str): 'cpu' or 'cuda'. The device on which to run the model.
        warmup_batches (int): Number of batches to run for GPU warm-up (for GPU only).
    
    Returns:
        avg_time_per_batch (float): Average inference time per batch.
    """
    model.eval()  # Set the model to evaluation mode

    total_time = 0
    num_batches = len(dataloader)
    
    # Warm-up for GPU
    if device == 'cuda':
        print("Warming up the GPU...")
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                
                if i >= warmup_batches:
                    break  # Only warm up for a few batches
                
                data = data.to(device)
                x_cont = data.x[:,:6]  # include puppi
                x_cat = data.x[:,6:].long()
                etaphi = torch.cat([data.x[:,3][:,None], data.x[:,4][:,None]], dim=1)
                edge_index = radius_graph(etaphi, r=0.4, batch=data.batch, loop=False, max_num_neighbors=255)  # turn off self-loop
                
                _ = model(x_cont, x_cat, edge_index, data.batch)
                
        torch.cuda.synchronize()  # Ensure all warm-up ops are done
        print("GPU warm-up complete.")

    # Timing inference
    print(f"Starting timing on {device.upper()}...")
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Inference Progress", leave=False):
            data = data.to(device)
            x_cont = data.x[:,:6]  # include puppi
            x_cat = data.x[:,6:].long()
            etaphi = torch.cat([data.x[:,3][:,None], data.x[:,4][:,None]], dim=1)
            edge_index = radius_graph(etaphi, r=0.4, batch=data.batch, loop=False, max_num_neighbors=255)  # turn off self-loop

            # Measure time
            if device == 'cuda':
                torch.cuda.synchronize()  # Sync before starting the timer
                start_time = time.time()
                
                _ = model(x_cont, x_cat, edge_index, data.batch)  # Inference
                
                torch.cuda.synchronize()  # Sync after model run
                end_time = time.time()
            else:
                # For CPU, just use time.time()
                start_time = time.time()
                
                _ = model(x_cont, x_cat, edge_index, data.batch)  # Inference
                
                end_time = time.time()

            # Accumulate total time
            total_time += (end_time - start_time)

    avg_time_per_batch = total_time / num_batches
    print(f"Total time on {device.upper()}: {total_time:.6f} seconds")
    print(f"Average time per batch on {device.upper()}: {avg_time_per_batch:.6f} seconds")
    
    return total_time, avg_time_per_batch

# Defining main function
def main():
    
    # ===================== INIT DATA LOADER ===================== #
    data_dir = '../../L1DeepMETv2/data_ttbar'
    dataloaders = data_loader.fetch_dataloader(data_dir = data_dir, batch_size=1, validation_split=.2)
    test_dl = dataloaders['test']
    print('Test dataloader: {}'.format(len(test_dl)))
    prefix = '../../L1DeepMETv2/ckpts_April30_scale_sigmoid'
    restore_ckpt = osp.join(prefix, 'last.pth.tar')
    
    # ===================== CPU TIMING ===================== #
    device = 'cpu'
    norm = torch.tensor([1., 1., 1., 1., 1., 1.]).to(device=device)
    torch_model = net.Net(continuous_dim=6, categorical_dim=2 , norm=norm).to(device)
    _ = utils.load_checkpoint(restore_ckpt, torch_model)
    print(f'Running inference on {device}')
    total_cpu_time, avg_cpu_time_per_batch = measure_inference_time(torch_model, test_dl, device=device)
    
    # ===================== GPU TIMING ===================== #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    norm = torch.tensor([1., 1., 1., 1., 1., 1.]).to(device=device)
    torch_model = net.Net(continuous_dim=6, categorical_dim=2 , norm=norm).to(device)
    _ = utils.load_checkpoint(restore_ckpt, torch_model)
    print(f'Running inference on {device}')
    total_gpu_time, avg_gpu_time_per_batch = measure_inference_time(torch_model, test_dl, device=device)
    
    results = {
        "Total CPU time": total_cpu_time,
        "Average CPU time per batch": avg_cpu_time_per_batch,
        "Total GPU time": total_gpu_time,
        "Average GPU time per batch": avg_gpu_time_per_batch if torch.cuda.is_available() else None
    }
    
    filename = "inference_times.txt"
    
    with open(filename, 'a') as f:  # Open the file in append mode
        f.write("Inference Results:\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.6f} seconds\n")
        f.write("\n")
    print(f"Results written to {filename}")


if __name__=="__main__":
    main()