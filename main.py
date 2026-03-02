import torch
import logging
import datetime
import numpy as np

from src.model_pinn import PINN
from src.train_pinn import train_pinn_brusselas
from src.process_data import ProcessDataBrusselas , ProcessDataColombia

logs_dir_path = Path().cwd() / "logs"
logs_dir_path.mkdir(parents=True, exist_ok=True)
log_file_path = logs_dir_path / f"train_pinn_{now()}.log"
#-------------------------------------------------------------------------------
from pathlib import Path

from config.basic_func import now
from config.logging_config import setup_production_config

from src.orchestrador import Orchestrador

Path().cwd() / 'config' / 'config.yaml'

def main():
    orchestrador = Orchestrador()

    pass


def main_old():
    
    lr = 1e-5
    R = 0.15
    num_epochs : int = 1000
    lamb = 1.0
    n_days = 93
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando dispositivo: {device}")
    
    data_path = Path().cwd() / "data" / "raw" / "em_cundinamarca_boyaca_251201_251231_11ws.parquet"
    data_path = Path().cwd() / "data" / "raw" / "weather_data.mat"
    data_path = Path().cwd() / "data" / "raw" / "weather_data.parquet"
    data_path = Path().cwd() / "data" / "raw" / "em_cundinamarca_boyaca_251201_251231_11ws_interpo.parquet"
    data_path = Path().cwd() / "data" / "raw" / "em_caribe3_20251001_20251231.parquet"
    data_path = Path().cwd() / "data" / "raw" / "em_caribe_20251201_20251231_ol.parquet"
    data_path = Path().cwd() / "data" / "raw" / "em_caribe_20251201_20251231.parquet"
    
    save_path = Path().cwd() / "models" / f"PINN_caribe3_epchos_{num_epochs}_lamb_{lamb}_R_{R}_days_{n_days}_{now()}.pth"
    
    model = PINN(input_dim=3, output_dim=3, hidden_neurons=600)
    logger.debug(model)

    # process_data = ProcessDataBrusselas(data_path)
    process_data = ProcessDataColombia(data_path)
    
    process_data.load_data()
    kwargs = {
        "R":R,
        "n_days":n_days,
        "interval":1,
        "WS_val_idx": np.array([4, 5, 9]),
        # "WS_val_idx": np.array([1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 16, 19]),
        }
    process_data.process_data(**kwargs)
    # print(f"{num_epochs=}")
    train_pinn_brusselas(process_data, model=model, device=device, lamb=lamb, num_epochs=num_epochs, save_path=save_path, lr=lr)

    ...


if __name__ == "__main__":
    setup_production_config(log_file_name=log_file_path)
    logger = logging.getLogger(__name__)
    try:
        main()
    except KeyboardInterrupt:
        print()
        logger.info('Proceso interrumpido manualmente')