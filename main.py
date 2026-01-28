from pathlib import Path
import numpy as np

from src.process_data import ProcessDataBrusselas
from src.train_pinn import train_pinn

data_path = Path().cwd() / "data" / "raw" / "weather_data.mat"
save_path = Path().cwd() / "models" / "PINN_brusselas_epchos_1000_lamb_2.0.pth"

def main():
    process_data = ProcessDataBrusselas(data_path)
    process_data.load_data()
    kwargs = {
        "interval":2,
        "WS_val_idx": np.array([1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 16, 19])}
    process_data.process_data(**kwargs)
    train_pinn(process_data, save_path=save_path)

    ...


if __name__=="__main__":
    main()