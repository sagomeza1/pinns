import torch
import numpy as np

from pathlib import Path
from src.model_pinn import PINN
from src.train_pinn import train_pinn_brusselas, train_pinn_colombia
from src.process_data import ProcessDataBrusselas , ProcessDataColombia

def main():
    
    num_epochs = 1000
    lamb = 2.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    # data_path = Path().cwd() / "data" / "raw" / "weather_data.mat"
    data_path = Path().cwd() / "data" / "raw" / "em_cundinamarca_251212_251231_full.parquet"
    save_path = Path().cwd() / "models" / "PINN_brusselas2_epchos_1000_lamb_2.0.pth"

    model = PINN(input_dim=3, output_dim=3, hidden_neurons=600)
    # process_data = ProcessDataBrusselas(data_path)
    process_data = ProcessDataColombia(data_path)
    process_data.load_data()
    kwargs = {
        "R":0.05,
        "n_days":14,
        "interval":1,
        "WS_val_idx": np.array([1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 16, 19])}
    process_data.process_data(**kwargs)
    train_pinn_brusselas(process_data, model=model, device=device, lamb=lamb, num_epochs=num_epochs, save_path=save_path)

    ...


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProceso interrupido manualmente.")