
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import scipy.io as sio

from .process_data import ProcessDataBrusselas as ProcessData
from .dataset import StationDataset, CollocationDataset
from .model_pinn import PINN
from .loss_functions import *


# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

def train_pinn(process_data: ProcessData, device=device, lamb:float = 2.0, num_epochs:int = 1000, save_path = Path('PINN_brusselas.pth')):
    # 1. Cargar datos
    print("Cargando y procesando datos...")
    # Asegúrate de tener el archivo .mat en la carpeta correcta o ajustar el path
    try:
        train_data, val_data, pinn_grid, params = process_data.return_data()
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return

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
    
    print(f"Batch Size Estaciones: {batch_WS}, Batch Size Grilla: {batch_PINN}")

    station_loader = DataLoader(station_dataset, batch_size=batch_WS, shuffle=True, drop_last=True)
    grid_loader = DataLoader(grid_dataset, batch_size=batch_PINN, shuffle=True, drop_last=True)

    # 3. Inicializar Modelo
    model = PINN(input_dim=3, output_dim=3, hidden_neurons=600).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    # Si la pérdida no baja en 15 épocas, reduce el LR a la mitad - Scheduler robusto (ReduceLROnPlateau).
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    # Listas para historial
    history = {'loss': [], 'ns_loss': [], 'p_loss': [], 'u_loss': [], 'v_loss': []}

    print("Iniciando entrenamiento...")
    
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
            
            loss_data = loss_u + loss_v + loss_p

            # 3. Pérdida Total (Suma ponderada compleja del original)
            # El original usa: (NS^2 + Data^2) / Sum(Losses). 
            # Esto es inusual, es una especie de normalización dinámica. Replicamos:
            total_sum = loss_physics + loss_data
            # final_loss = (loss_physics**2 + loss_u**2 + loss_v**2 + loss_p**2) / total_sum
            final_loss = total_sum

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
        
        # Actualizar el scheduler basado en la pérdida promedio
        scheduler.step(avg_loss)
        
        current_lr = optimizer.param_groups[0]['lr']        
        
        # Ajuste adaptativo de Learning Rate (Lógica manual original)
        if avg_loss > 1e-1:
            lr = 1e-3
        elif avg_loss > 3e-2:
            lr = 1e-4
        elif avg_loss > 3e-3:
            lr = 1e-5
        else:
            lr = 1e-6
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Guardar historial
        history['loss'].append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} Loss: {avg_loss:.3e} LR: {current_lr:.1e}")

        # Guardado periódico
        if (epoch + 1) % num_epochs == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Modelo guardado en: {save_path}")

            # Aquí podrías agregar la lógica de inferencia sobre X_PINN y guardar el .mat
            # similar al bloque 'Save Data' del original, usando model.eval()

    print("Proceso completado.")