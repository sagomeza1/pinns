import torch
import numpy as np

from pathlib import Path
from src.model_pinn import PINN
from src.train_pinn import train_pinn_brusselas, train_pinn_colombia
from src.process_data import ProcessDataBrusselas , ProcessDataColombia

def main():
    
    num_epochs = 30
    lamb = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    data_path = Path().cwd() / "data" / "raw" / "em_cundinamarca_boyaca_251201_251231_11ws.parquet"
    data_path = Path().cwd() / "data" / "raw" / "weather_data.mat"
    data_path = Path().cwd() / "data" / "raw" / "em_cundinamarca_boyaca_251201_251231_11ws_interpo.parquet"
    
    save_path = Path().cwd() / "models" / f"PINN_cunboy_epchos_{num_epochs}_lamb_{lamb}.pth"
    
    model = PINN(input_dim=3, output_dim=3, hidden_neurons=600)
    # print(model)

    process_data = ProcessDataBrusselas(data_path)
    process_data = ProcessDataColombia(data_path)
    
    process_data.load_data()
    kwargs = {
        # "R":0.15,
        "n_days":30,
        "interval":1,
        "WS_val_idx": np.array([4, 8])}
        # "WS_val_idx": np.array([1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 16, 19])}
    process_data.process_data(**kwargs)
    train_pinn_brusselas(process_data, model=model, device=device, lamb=lamb, num_epochs=num_epochs, save_path=save_path)

    ...


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProceso interrupido manualmente.")