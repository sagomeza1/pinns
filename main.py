from pathlib import Path
import numpy as np

from src.process_data import ProcessDataBrusselas , ProcessDataColombia
from src.train_pinn import train_pinn

# data_path_bru = Path().cwd() / "data" / "raw" / "weather_data.mat"
data_path_cun = Path().cwd() / "data" / "raw" / "em_cundinamarca_251212_251231_full.parquet"
save_path = Path().cwd() / "models" / "PINN_cundinamarca_epchos_1000_lamb_2.0.pth"

def main():
    # process_data = ProcessDataBrusselas(datas_path_bru)
    process_data = ProcessDataColombia(data_path_cun)
    process_data.load_data()
    kwargs = {
        "interval":1,
        "WS_val_idx": np.array([1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 16, 19])}
    process_data.process_data(**kwargs)
    train_pinn(process_data, save_path=save_path)

    ...


if __name__=="__main__":
    main()