import sys
sys.path.append("../config")

import torch
import logging
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import scipy.io as sio
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from .process_data import ProcessDataBrusselas as ProcessData
from .dataset import StationDataset, CollocationDataset
from .loss_functions import *

logger = logging.getLogger(__name__)
mse = nn.MSELoss()

# Configuración del dispositivo

def train_pinn_brusselas(process_data: ProcessData, model:nn.Module, device:str, lr:float = 1e-5, lamb:float = 2.0, num_epochs:int = 1000, save_path = Path('PINN_brusselas.pth')):
    # 1. Cargar datos
    # Asegúrate de tener el archivo .mat en la carpeta correcta o ajustar el path
    try:
        train_data, val_data, pinn_grid, params = process_data.return_data()
        logger.info("Datos cargados.")
    except Exception as e:
        print(f'{" ERROR":.>20}')
        print(f"Error cargando datos: {e}")
        return None
    
    logger.info(f"Número de epocas: {num_epochs:,}.")
    logger.debug(f"{lamb=}.")

    # 2. Preparar Datasets y DataLoaders
    station_dataset = StationDataset(train_data)
    grid_dataset = CollocationDataset(pinn_grid)
    
    # Cálculos de tamaño de lote (replican lógica original)
    batch_WS = int(np.ceil(len(station_dataset) / params['n_days'] * params['R']))
    batch_PINN = int(np.ceil(len(grid_dataset) / params['n_days'] * params['R']))
    
    logger.info(f"Registro estaciones: {int(len(station_dataset)):,}.")
    logger.info(f"Grilla: {int(len(grid_dataset)):,}.")

    station_loader = DataLoader(station_dataset, batch_size=batch_WS, shuffle=True, drop_last=True)
    grid_loader = DataLoader(grid_dataset, batch_size=batch_PINN, shuffle=True, drop_last=True)
    logger.info(f"Batch estaciones: {batch_WS}.")
    logger.info(f"Batch grilla: {batch_PINN}.")

    # 3. Inicializar Modelo
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.Adam(model.parameters(), lr=1e-5)
    logger.debug(f"{optimizer=}")
    
    # Si la pérdida no baja en 15 épocas, reduce el LR a la mitad - Scheduler robusto (ReduceLROnPlateau).
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3162, patience=50, threshold=1e-2)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-2)    
    logger.debug(f"{scheduler.mode=}")
    logger.debug(f"{scheduler.factor=}")
    logger.debug(f"{scheduler.threshold_mode=}")
    logger.debug(f"{scheduler.threshold=}")
    logger.debug(f"{scheduler.patience=}")
    logger.debug(f"{scheduler.cooldown=}")
    logger.debug(f"{scheduler.min_lrs=}")
    
    # Listas para historial
    history = {'epoch': [],'loss': [], 'ns_loss': [], 'p_loss': [], 'u_loss': [], 'v_loss': [], 'lr': []}
    
    pre_lr = 0.0

    logger.info("Iniciando entrenamiento")
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
            
            optimizer.zero_grad()
            
            # 1. Pérdida Física (NS equations)
            loss_ns_grid = loss_navier_stokes(model, t_f, x_f, y_f)
            loss_ns_data = loss_navier_stokes(model, t_u, x_u, y_u)
            loss_physics = lamb * (loss_ns_grid + loss_ns_data)

            # 2. Pérdida de Datos (Predicción vs Real)
            # Hacemos forward pass para datos
            out_data = model(t_u, x_u, y_u)
            u_pred, v_pred, p_pred = out_data[:, 0:1], out_data[:, 1:2], out_data[:, 2:3]
            
            loss_u = loss_data_variable(u_pred, u_true)
            loss_v = loss_data_variable(v_pred, v_true)
            loss_p = loss_data_variable(p_pred, p_true)
            
            final_loss = (loss_physics**2 + loss_u**2 + loss_v**2 + loss_p**2) / (loss_physics + loss_u + loss_v + loss_p)

            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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
        if avg_ns < 1e-5:
            logger.warning(f"Loss NS demasiado bajo: {avg_ns:.3e}.")
                
        # logger.debug(f"Número de batches: {batches}.")

        # Actualizar el scheduler basado en la pérdida promedio
        scheduler.step(avg_loss)
        
        # current_lr
        lr = optimizer.param_groups[0]['lr']        
        if pre_lr != lr:
            logger.info(f"Cambio de lr: {pre_lr:.1e} -> {lr:.1e}.")
        pre_lr = lr
        
        # Guardar historial
        history['epoch'].append(epoch)
        history['loss'].append(float(avg_loss) if isinstance(avg_loss, torch.Tensor) else avg_loss)
        history['ns_loss'].append(float(avg_ns) if isinstance(avg_ns, torch.Tensor) else avg_ns)
        history['p_loss'].append(float(avg_data_p) if isinstance(avg_data_p, torch.Tensor) else avg_data_p)
        history['u_loss'].append(float(avg_data_u) if isinstance(avg_data_u, torch.Tensor) else avg_data_u)
        history['v_loss'].append(float(avg_data_v) if isinstance(avg_data_v, torch.Tensor) else avg_data_v)
        history['lr'].append(float(lr) if isinstance(lr, torch.Tensor) else lr)
        
        if epoch % 10 == 0:
            logger.info(f"| Epoch: {epoch:4} | Loss: {avg_loss:.3e} | LR: {lr:.1e} |")
        logger.debug(f"|Epoch: {epoch:4}|Loss: {avg_loss:.3e}|Loss ns: {avg_ns:.3e}|Loss u: {avg_data_u:.3e}|Loss v: {avg_data_v:.3e}|Loss p: {avg_data_p:.3e}|LR: {lr:.1e}|")

        # Guardado periódico
        if (epoch + 1) % (num_epochs // 4) == 0:
            epoch_save_path = save_path.with_name(f"{save_path.stem}_epoch_{epoch + 1}").with_suffix(save_path.suffix)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, epoch_save_path)
            logger.info(f"Modelo guardado en: {epoch_save_path}.")
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
            logger.info(f"Métricas almacendas en : {metrics_path}.")
            logger.info(f"En la epoca {epoch + 1} con loss {avg_loss:.2e}.")

    logger.info("Proceso completado.")

