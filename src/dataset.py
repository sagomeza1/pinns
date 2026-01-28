import torch
from torch.utils.data import Dataset

class StationDataset(Dataset):
    """Dataset para los datos medidos en estaciones (u, v, p conocidos)."""
    def __init__(self, data_dict):
        # Aplanar los datos: (N_stations * Time_steps, 1)
        self.t = torch.tensor(data_dict['T'].flatten(), dtype=torch.float32).unsqueeze(1)
        self.x = torch.tensor(data_dict['X'].flatten(), dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(data_dict['Y'].flatten(), dtype=torch.float32).unsqueeze(1)
        self.u = torch.tensor(data_dict['U'].flatten(), dtype=torch.float32).unsqueeze(1)
        self.v = torch.tensor(data_dict['V'].flatten(), dtype=torch.float32).unsqueeze(1)
        self.p = torch.tensor(data_dict['P'].flatten(), dtype=torch.float32).unsqueeze(1)
        
        # Filtrar NaNs globales (si existen en cualquiera de las variables target)
        # Nota: El código original trata u, v, p por separado. Aquí simplificamos 
        # asumiendo que si falta u, falta v. Si no, se pueden crear máscaras.
        mask = ~torch.isnan(self.u.squeeze()) & ~torch.isnan(self.v.squeeze()) & ~torch.isnan(self.p.squeeze())
        self.t = self.t[mask]
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.u = self.u[mask]
        self.v = self.v[mask]
        self.p = self.p[mask]

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return self.t[idx], self.x[idx], self.y[idx], self.u[idx], self.v[idx], self.p[idx]

class CollocationDataset(Dataset):
    """Dataset para puntos de colocación (grilla PINN) donde se evalúa la física."""
    def __init__(self, grid_dict):
        self.t = torch.tensor(grid_dict['T'].flatten(), dtype=torch.float32).unsqueeze(1)
        self.x = torch.tensor(grid_dict['X'].flatten(), dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(grid_dict['Y'].flatten(), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return self.t[idx], self.x[idx], self.y[idx]
