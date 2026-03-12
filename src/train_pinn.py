
import torch
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import scipy.io as sio
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from pathlib import Path

from .process_data import ProcessDataBrusselas as ProcessData
from .dataset import StationDataset, CollocationDataset
from .loss_functions import *

mse = nn.MSELoss()

# Configuración del dispositivo

def now():
    t0 = datetime.datetime.now()
    return f"{str(t0.year)[-2:]}{t0.month:02}{t0.day:02}{t0.hour:02}{t0.minute:02}"

def train_pinn_brusselas(process_data: ProcessData, model:nn.Module, device:str, lamb:float = 2.0, num_epochs:int = 1000, save_path = Path('PINN_brusselas.pth')):
    # 1. Cargar datos
    print("-"*70)
    print(f'{"ENTRENAMIENTO CON ADAM":^70}')
    print(f'{"Cargando y procesando datos ":.<50}', end="")
    try:
        train_data, val_data, pinn_grid, params = process_data.return_data()
        print(f'{" OK":.>20}')
    except Exception as e:
        print(f'{" ERROR":.>20}')
        print(f"Error cargando datos: {e}")
        return
    print("-"*70)

    adam_epochs = num_epochs
    
    print(f'{"Total de épocas: ":<50}{num_epochs:>20,}')
    print(f'{"Épocas con Adam: ":<50}{adam_epochs:>20,}')
    print(f'{"lambda (física): ":<50}{lamb:>20,}')

    # 2. Preparar Datasets y DataLoaders
    station_dataset = StationDataset(train_data)
    grid_dataset = CollocationDataset(pinn_grid)
    
    batch_WS = int(np.ceil(len(station_dataset) / params['n_days'] * params['R']))
    batch_PINN = int(np.ceil(len(grid_dataset) / params['n_days'] * params['R']))
    
    print(f'{"Registros estaciones: ":<50}{len(station_dataset):>20,}')
    print(f'{"Grilla PINN: ":<50}{len(grid_dataset):>20,}')
    print(f'{"Batch estaciones (Adam): ":<50}{batch_WS:>20,}')
    print(f'{"Batch grilla (Adam): ":<50}{batch_PINN:>20,}')
    print("-"*70)

    station_loader = DataLoader(station_dataset, batch_size=batch_WS, shuffle=True, drop_last=True)
    grid_loader = DataLoader(grid_dataset, batch_size=batch_PINN, shuffle=True, drop_last=True)

    # 3. Inicializar Modelo y Optimizadores
    model.to(device)
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, mode='min', factor=0.5, patience=50, threshold=1e-3)
    history = {'epoch': [],'loss': [], 'ns_loss': [], 'p_loss': [], 'u_loss': [], 'v_loss': [], 'lr': [], }

    print(f" Fase 1: Entrenamiento con Adam en {str(device).upper()} ".center(70, "="))
    
    for epoch in range(adam_epochs):
        model.train()
        
        epoch_loss, epoch_ns, epoch_data_p, epoch_data_u, epoch_data_v, batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0

        for (batch_st, batch_gr) in zip(station_loader, grid_loader):
            t_u, x_u, y_u, u_true, v_true, p_true = [b.to(device) for b in batch_st]
            t_f, x_f, y_f = [b.to(device) for b in batch_gr]
            
            optimizer_adam.zero_grad()
            
            loss_ns_grid = loss_navier_stokes(model, t_f, x_f, y_f)
            loss_ns_data = loss_navier_stokes(model, t_u, x_u, y_u)
            loss_physics = lamb * (loss_ns_grid + loss_ns_data)

            out_data = model(t_u, x_u, y_u)
            u_pred, v_pred, p_pred = out_data[:, 0:1], out_data[:, 1:2], out_data[:, 2:3]
            
            loss_u = loss_data_variable(u_pred, u_true)
            loss_v = loss_data_variable(v_pred, v_true)
            loss_p = loss_data_variable(p_pred, p_true)
            
            final_loss = (loss_physics**2 + loss_u**2 + loss_v**2 + loss_p**2) / (loss_physics + loss_u + loss_v + loss_p)
            
            # with torch.no_grad():
            #     denom = (loss_physics + loss_u + loss_v + loss_p)
            # final_loss = (loss_physics**2 + loss_u**2 + loss_v**2 + loss_p**2) / denom if denom > 1e-9 else (loss_physics + loss_u + loss_v + loss_p)

            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer_adam.step()

            epoch_loss += final_loss.item()
            epoch_ns += loss_physics.item()
            epoch_data_u += loss_u.item()
            epoch_data_v += loss_v.item()
            epoch_data_p += loss_p.item()
            batches += 1

        avg_loss = epoch_loss / batches
        avg_ns = epoch_ns / batches
        avg_data_u = epoch_data_u / batches
        avg_data_v = epoch_data_v / batches
        avg_data_p = epoch_data_p / batches
        
        scheduler.step(avg_loss)
        current_lr = optimizer_adam.param_groups[0]['lr']
        
        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        history['ns_loss'].append(avg_ns)
        history['p_loss'].append(avg_data_p)
        history['u_loss'].append(avg_data_u)
        history['v_loss'].append(avg_data_v)
        history['lr'].append(current_lr)
        
        if epoch % 10 == 0:
            print(f"| ADAM | Epoch: {epoch:4} | Loss: {avg_loss:.3e} | NS: {avg_ns:.3e} | U: {avg_data_u:.3e} | V: {avg_data_v:.3e} | P: {avg_data_p:.3e} | LR: {current_lr:.1e} |")

        if (epoch + 1) % (num_epochs // 8) == 0:
            torch.save(model.state_dict(), save_path.with_name(f"{save_path.stem}_{epoch + 1}").with_suffix(save_path.suffix))

    final_save_path = save_path.with_name(f"{save_path.stem}_final").with_suffix(save_path.suffix)
    torch.save(model.state_dict(), final_save_path)
    print(f"Modelo final guardado en: {final_save_path}")

    metrics_path = save_path.with_name(f"history_{save_path.stem}").with_suffix(".mat")
    sio.savemat(metrics_path, history)
    print(f"Métricas finales almacenadas en: {metrics_path}")

    print("Proceso completado.")
    
    
#%%

1+1
# %%
