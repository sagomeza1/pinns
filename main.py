import torch
import datetime
import numpy as np

from pathlib import Path
from src.model_pinn import PINN
from src.train_pinn import train_pinn_brusselas
from src.process_data import ProcessDataBrusselas , ProcessDataColombia

def now():
    t0 = datetime.datetime.now()
    return f"{str(t0.year)[-2:]}{t0.month:02}{t0.day:02}{t0.hour:02}{t0.minute:02}"

def main():
    
    num_epochs : int = 2000
    lamb = 3.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    data_path = Path().cwd() / "data" / "raw" / "em_cundinamarca_boyaca_251201_251231_11ws.parquet"
    data_path = Path().cwd() / "data" / "raw" / "weather_data.mat"
    data_path = Path().cwd() / "data" / "raw" / "weather_data.parquet"
    data_path = Path().cwd() / "data" / "raw" / "em_cundinamarca_boyaca_251201_251231_11ws_interpo.parquet"
    data_path = Path().cwd() / "data" / "raw" / "em_caribe_20251201_20251231.parquet"
    
    save_path = Path().cwd() / "models" / f"PINN_caribe_epchos_{num_epochs}_lamb_{lamb}_{now()}.pth"
    
    model = PINN(input_dim=3, output_dim=3, hidden_neurons=600)
    # print(model)

    # process_data = ProcessDataBrusselas(data_path)
    process_data = ProcessDataColombia(data_path)
    
    process_data.load_data()
    kwargs = {
        "R":0.1,
        "n_days":30,
        "interval":1,
        "WS_val_idx": np.array([4, 8, 16, 18]),
        # "WS_val_idx": np.array([1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 16, 19]),
        }
    process_data.process_data(**kwargs)
    process_data.resume()
    print(f"{num_epochs=}")
    train_pinn_brusselas(process_data, model=model, device=device, lamb=lamb, num_epochs=num_epochs, save_path=save_path)

    ...


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProceso interrupido manualmente.")