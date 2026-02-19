
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
    print(f'{"ENTRENAMIENTO":^70}')
    print(f'{"Cargando y procesando datos ":.<50}', end="")
    # Asegúrate de tener el archivo .mat en la carpeta correcta o ajustar el path
    try:
        train_data, val_data, pinn_grid, params = process_data.return_data()
        print(f'{" OK":.>20}')
    except Exception as e:
        print(f'{" ERROR":.>20}')
        print(f"Error cargando datos: {e}")
        return
    print("-"*70)
    print(f'{"Epocas: ":<50}{num_epochs:>20,}')
    print(f'{"lamb: ":<50}{lamb:>20,}')

    # 2. Preparar Datasets y DataLoaders
    # Dataset de Estaciones (Datos observados)
    station_dataset = StationDataset(train_data)
    
    # Dataset de Colocación (Grilla para física + puntos de estaciones para física también)
    # El original usa t_eqns (PINN grid) y t_eqns_ref (Estaciones grid).
    # Combinaremos ambos para la física.
    grid_dataset = CollocationDataset(pinn_grid)
    
    # Cálculos de tamaño de lote (replican lógica original)
    batch_WS = int(np.ceil(len(station_dataset) / params['n_days'] * params['R']))
    batch_PINN = int(np.ceil(len(grid_dataset) / params['n_days'] * params['R']))
    
    print(f'{"Registros estaciones: ":<50}{int(len(station_dataset)):>20,}')
    print(f'{"Grilla: ":<50}{int(len(grid_dataset)):>20,}')

    print(f'{"Batch estaciones: ":<50}{batch_WS:>20,}')
    print(f'{"Batch grilla: ":<50}{batch_PINN:>20,}')
    
    print("-"*70)

    station_loader = DataLoader(station_dataset, batch_size=batch_WS, shuffle=True, drop_last=True)
    grid_loader = DataLoader(grid_dataset, batch_size=batch_PINN, shuffle=True, drop_last=True)

    # 3. Inicializar Modelo
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    # optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    # Si la pérdida no baja en 15 épocas, reduce el LR a la mitad - Scheduler robusto (ReduceLROnPlateau).
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-3)
    
    # Listas para historial
    history = {'epoch': [],'loss': [], 'ns_loss': [], 'p_loss': [], 'u_loss': [], 'v_loss': [], 'lr': []}

    print()
    print("Iniciando entrenamiento")
    
    for epoch in range(num_epochs):
        model.train()
        
        epoch_loss = 0.0
        epoch_ns = 0.0
        epoch_data_p = 0.0
        epoch_data_u = 0.0
        epoch_data_v = 0.0
        batches = 0

        # Iterar sobre data loaders. Usamos zip, aunque tengan longitudes distintas (corta en el menor)
        # Ojo: Para entrenamiento robusto, se suele usar itertools.cycle en el más corto.
        for (batch_st, batch_gr) in zip(station_loader, grid_loader):
            
            # Datos Observados
            t_u, x_u, y_u, u_true, v_true, p_true = [b.to(device) for b in batch_st]
            
            # Datos de Colocación (Física)
            t_f, x_f, y_f = [b.to(device) for b in batch_gr]
            
            # También usamos los puntos de estaciones para la física (como en el original 'NS_data')
            t_f_ref, x_f_ref, y_f_ref = t_u, x_u, y_u

            optimizer.zero_grad()
            
            # 1. Pérdida Física (NS equations)
            loss_ns_grid = loss_navier_stokes(model, t_f, x_f, y_f)
            loss_ns_data = loss_navier_stokes(model, t_f_ref, x_f_ref, y_f_ref)
            loss_physics = lamb * (loss_ns_grid + loss_ns_data)

            # 2. Pérdida de Datos (Predicción vs Real)
            # Hacemos forward pass para datos
            out_data = model(t_u, x_u, y_u)
            u_pred, v_pred, p_pred = out_data[:, 0:1], out_data[:, 1:2], out_data[:, 2:3]
            
            loss_u = loss_data_variable(u_pred, u_true)
            loss_v = loss_data_variable(v_pred, v_true)
            loss_p = loss_data_variable(p_pred, p_true)
            
            # loss_data = loss_u + loss_v + loss_p

            # 3. Pérdida Total (Suma ponderada compleja del original)
            # El original usa: (NS^2 + Data^2) / Sum(Losses). 
            # Esto es inusual, es una especie de normalización dinámica. Replicamos:
            # total_sum = loss_physics + loss_data
            # final_loss = (loss_physics**2 + loss_u**2 + loss_v**2 + loss_p**2) / total_sum
            
            # final_loss = total_sum
            final_loss = (loss_physics**2 + loss_u**2 + loss_v**2 + loss_p**2) / (loss_physics + loss_u + loss_v + loss_p)

            final_loss.backward()
            
            # CAMBIO 4: Gradient Clipping
            # Esto evita que un gradiente explosivo rompa los pesos y cause el salto a 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Acumular métricas
            epoch_loss += final_loss.item()
            epoch_ns += loss_physics.item()
            epoch_data_u += loss_u.item()
            epoch_data_v += loss_v.item()
            epoch_data_p += loss_p.item()
            batches += 1

        # Promedios
        avg_loss = epoch_loss / batches
        avg_ns = epoch_ns / batches
        avg_data_u = epoch_data_u / batches
        avg_data_v = epoch_data_v / batches
        avg_data_p = epoch_data_p / batches
        
        # Actualizar el scheduler basado en la pérdida promedio
        scheduler.step(avg_loss)
        
        current_lr = optimizer.param_groups[0]['lr']        
        
        # Ajuste adaptativo de Learning Rate (Lógica manual original)
        if avg_loss > 1e-1:
            lr = 1e-5
        elif avg_loss > 3e-2:
            lr = 1e-6
        elif avg_loss > 3e-3:
            lr = 1e-7
        else:
            lr = 1e-8
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Guardar historial
        history['epoch'].append(epoch)
        history['loss'].append(float(avg_loss) if isinstance(avg_loss, torch.Tensor) else avg_loss)
        history['ns_loss'].append(float(avg_ns) if isinstance(avg_ns, torch.Tensor) else avg_ns)
        history['p_loss'].append(float(avg_data_p) if isinstance(avg_data_p, torch.Tensor) else avg_data_p)
        history['u_loss'].append(float(avg_data_u) if isinstance(avg_data_u, torch.Tensor) else avg_data_u)
        history['v_loss'].append(float(avg_data_v) if isinstance(avg_data_v, torch.Tensor) else avg_data_v)
        history['lr'].append(float(lr) if isinstance(lr, torch.Tensor) else lr)
        
        if epoch % 10 == 0:
            print(f"| Epoch: {epoch:4} | Loss: {avg_loss:.3e} | Loss ns: {avg_ns:.3e} | Loss u: {avg_data_u:.3e} | Loss v: {avg_data_v:.3e} | Loss p: {avg_data_p:.3e} | LR: {current_lr:.1e} |")

        # Guardado periódico
        if (epoch + 1) % num_epochs == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Modelo guardado en: {save_path}")

            metrics_path = save_path.with_name(f"history_{save_path.stem}").with_suffix(".mat")
            # Guardando métricas del modelo
            sio.savemat(metrics_path,
                        {
                            "epoch": history["epoch"],
                            "loss": history["loss"],
                            "ns_loss": history["ns_loss"],
                            "u_loss": history["u_loss"],
                            "v_loss": history["v_loss"],
                            "p_loss": history["p_loss"],
                            "lr": history["lr"],
                        })
            print(f"Métricas almacendas en : {metrics_path}")

    print("Proceso completado.")