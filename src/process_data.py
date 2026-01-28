import os
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as sio
import datetime as dt
import matplotlib.pyplot as plt

from datetime import datetime
from typing import Optional, Any

class ProcessDataBrusselas:
    '''
    Se crea la clase ProcessData para realizar el procesamiento
    de los datos, considerando ciertas especificaciones.

    n_días: Cantidad de días
    n_estaciones: Cantidad de estaciones
    delta_time: Diferencia de tiempo registrada
    '''
    def __init__(self, filepath:Path):
        # Ruta del archivo
        self.filepath = filepath

        # Tiempos registados
        self._fecha_creacion = datetime.now()
        self._tiempo_de_carga: Optional[Any] = None

        # Verificación de la data procesada
        self._state_data_process = False

        # dictionary to record the parameneters
        self.params = dict()

    def load_data(self) -> None:
        '''
        Se genera la carga de los datos
        '''
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"No se encontró el archivo: {self.filepath}")
        print(f"Iniciando carga de {self.filepath}")
        t0 = datetime.now()
        self.WS_data = sio.loadmat(self.filepath)
        self._tiempo_de_carga = datetime.now() - t0

    def process_data(self, **kwargs) -> None:

        print("--- Conversión de fecha a tiempo continuo (segundos) ---")
        self._process_time()
        print("--- OK ---")

        print("--- Coordenadas Cartesianas y Proyecciones ---")
        self._process_coordinates_and_projections()
        print("--- OK ---")

        print("--- Eliminar NaNs ---")
        self._delete_nans()
        print("--- OK ---")

        print("--- Selección de días y ordenamiento por coor ---")
        # Filtrar kwargs: solo pasar los que acepta _filter_by_time_and_sort_by_coor
        filter_kwargs = {k: v for k, v in kwargs.items() if k in ['n_days', 'interval']}
        self._filter_by_time_and_sort_by_coor(**filter_kwargs)
        print("--- OK ---")

        print("--- Corrección de presión a nivel del mar (ISA) ---")
        # Corrección de presión a nivel del mar (ISA)
        self.P_WS = self.P_WS * (1 - 0.0065 * self.Z_WS / (self.Temp_WS + 273.15 + 0.0065 * self.Z_WS))**(-5.257)
        print("--- OK ---")

        print("--- Centrado y creación de la malla ---")
        self._centered_grid_adimensionalization()
        print("--- OK ---")

        print("--- Separar entre validación y entrenamiento ---")
        # Filtrar kwargs: solo pasar los que acepta _split_validation_train
        split_kwargs = {k: v for k, v in kwargs.items() if k in ['WS_val_idx']}
        self._split_validation_train(**split_kwargs)
        print("--- OK ---")

        self._state_data_process = True

    def _process_time(self) -> None:
        # Conversión de fecha a tiempo continuo (segundos)
        date_0 = self.WS_data['Date'][0]
        date = []
        for i in range(len(date_0)):
            date = np.append(date, str(date_0[i])[2:-2])

        # Manejo inicial de NaNs en fechas
        time_init = dt.datetime(int(date[0][0:4]), int(date[0][5:7]), int(date[0][8:10]),
                                int(date[0][11:13]), int(date[0][14:16]))
        self._T_nan_index = np.argwhere(pd.isna(date))
        date = np.delete(date, self._T_nan_index[:, 0], 0)

        Seconds = np.zeros((date.shape[0], 1))
        for index in range(date.shape[0]):
            d = dt.datetime(int(date[index][0:4]), int(date[index][5:7]), int(date[index][8:10]),
                            int(date[index][11:13]), int(date[index][14:16]))
            Seconds[index, 0] = (d - time_init).total_seconds()

        self.T_WS = Seconds

    def _process_coordinates_and_projections(self) -> None:
        # Coordenadas Cartesianas y Proyecciones
        self.X_WS = np.array(6378000 * np.sin(np.radians(self.WS_data['Lon'])))[0]
        self.Y_WS = np.array(6378000 * np.sin(np.radians(self.WS_data['Lat'])))[0]
        self.Z_WS = np.array(self.WS_data['Alt'])[0]
        self.Temp_WS = np.array(self.WS_data['Temperature'])[0]
        self.U_WS = (self.WS_data['WindSpeed'] * self.WS_data['WindDirectionX'])[0]
        self.V_WS = (self.WS_data['WindSpeed'] * self.WS_data['WindDirectionY'])[0]
        self.P_WS = self.WS_data['Pressure'][0] * 100 # mbar a Pa

    def _delete_nans(self) -> None:
        # Eliminar NaNs de campos temporales
        for arr in [self.X_WS, self.Y_WS, self.Z_WS, self.U_WS,
                    self.V_WS, self.P_WS, self.Temp_WS]:
            arr = np.delete(arr, self._T_nan_index[:, 0], 0) # type: ignore

        self.T_WS = self._reshape_data(self.T_WS)
        self.X_WS = self._reshape_data(self.X_WS)
        self.Y_WS = self._reshape_data(self.Y_WS)
        self.Z_WS = self._reshape_data(self.Z_WS)
        self.U_WS = self._reshape_data(self.U_WS)
        self.V_WS = self._reshape_data(self.V_WS)
        self.P_WS = self._reshape_data(self.P_WS)
        self.Temp_WS = self._reshape_data(self.Temp_WS)

        # Eliminar estaciones con NaNs en ubicación
        X_nan_index = np.argwhere(np.isnan(self.X_WS))
        # Nota: Se asume eliminación simétrica en todas las variables
        self.T_WS = np.delete(self.T_WS, X_nan_index[:, 0], 0)
        self.P_WS = np.delete(self.P_WS, X_nan_index[:, 0], 0)
        self.U_WS = np.delete(self.U_WS, X_nan_index[:, 0], 0)
        self.V_WS = np.delete(self.V_WS, X_nan_index[:, 0], 0)
        self.X_WS = np.delete(self.X_WS, X_nan_index[:, 0], 0)
        self.Y_WS = np.delete(self.Y_WS, X_nan_index[:, 0], 0)
        self.Z_WS = np.delete(self.Z_WS, X_nan_index[:, 0], 0)
        self.Temp_WS = np.delete(self.Temp_WS, X_nan_index[:, 0], 0)

    def _reshape_data(self, arr: np.ndarray, num_stations: int = 21) -> None:
        # Reestructurar matrices: 21 estaciones x mediciones
        return np.reshape(arr, (int(arr.shape[0] / num_stations), num_stations)).T

    def _filter_by_time_and_sort_by_coor(self, n_days: int = 14, interval: int = 1) -> None:
        """
        interval is the value to indicate the time difference between records.
        1 indicate 10 min
        2 indicate 20 min
        .
        .
        .
        6 indicate 60 min or 1h
        """
        # Selection of days
        samples = int(144 * n_days)

        self.T_WS = self.T_WS[:, :samples]
        self.X_WS = self.X_WS[:, :samples]
        self.Y_WS = self.Y_WS[:, :samples]
        self.Z_WS = self.Z_WS[:, :samples]
        self.U_WS = self.U_WS[:, :samples]
        self.V_WS = self.V_WS[:, :samples]
        self.P_WS = self.P_WS[:, :samples]
        self.Temp_WS = self.Temp_WS[:, :samples]

        self.params["n_days"] = n_days
        print(f"{n_days} selected days")

        # Selection of interval
        self.T_WS = self.T_WS[:, ::interval]
        self.X_WS = self.X_WS[:, ::interval]
        self.Y_WS = self.Y_WS[:, ::interval]
        self.Z_WS = self.Z_WS[:, ::interval]
        self.U_WS = self.U_WS[:, ::interval]
        self.V_WS = self.V_WS[:, ::interval]
        self.P_WS = self.P_WS[:, ::interval]
        self.Temp_WS = self.Temp_WS[:, ::interval]

        self.params["interval"] = interval
        print(f"Interval of {interval * 10} min")


        # Order by coor X
        for snap in range(self.T_WS.shape[1]):
            idx_sort = np.argsort(self.X_WS[:, snap])
            self.T_WS[:, snap] = self.T_WS[idx_sort, snap]
            self.X_WS[:, snap] = self.X_WS[idx_sort, snap]
            self.Y_WS[:, snap] = self.Y_WS[idx_sort, snap]
            self.Z_WS[:, snap] = self.Z_WS[idx_sort, snap]
            self.U_WS[:, snap] = self.U_WS[idx_sort, snap]
            self.V_WS[:, snap] = self.V_WS[idx_sort, snap]
            self.P_WS[:, snap] = self.P_WS[idx_sort, snap]
            self.Temp_WS[:, snap] = self.Temp_WS[idx_sort, snap]

    def _centered_grid_adimensionalization(self) -> None:
        # Centrado
        x_min, x_max = np.min(self.X_WS), np.max(self.X_WS)
        y_min, y_max = np.min(self.Y_WS), np.max(self.Y_WS)
        t_min, t_max = np.min(self.T_WS), np.max(self.T_WS)

        self.X_WS = self.X_WS - (x_min + x_max) / 2
        self.Y_WS = self.Y_WS - (y_min + y_max) / 2
        self.T_WS = self.T_WS - t_min

        # Grilla PINN
        T_PINN = self.T_WS[0:1, :]
        R = 0.2 # Grados
        R_PINN = 6378000 * np.sin(np.radians(R))
        x_PINN = np.arange(x_min - R_PINN, x_max + R_PINN, R_PINN) - (x_min + x_max) / 2
        y_PINN = np.arange(y_min - R_PINN, y_max + R_PINN, R_PINN) - (y_min + y_max) / 2

        X_PINN_grid, Y_PINN_grid = np.meshgrid(x_PINN, y_PINN)
        X_PINN_flat = X_PINN_grid.flatten('F')[:, None]
        Y_PINN_flat = Y_PINN_grid.flatten('F')[:, None]

        dim_T_PINN = T_PINN.shape[1]
        dim_N_PINN = X_PINN_flat.shape[0]

        T_PINN_full = np.tile(T_PINN, (dim_N_PINN, 1))
        X_PINN_full = np.tile(X_PINN_flat, dim_T_PINN)
        Y_PINN_full = np.tile(Y_PINN_flat, dim_T_PINN)

        # Adimensionalización
        L = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        W = np.sqrt(np.nanmax(np.abs(self.U_WS))**2 + np.nanmax(np.abs(self.V_WS))**2)
        rho = 1.269
        nu = 1.382e-5
        Re = int(W * L / nu)
        P0 = np.nanmean(self.P_WS)

        print(f'-- L: {L:.2f}, W: {W:.2f}, P0: {P0:.2f}, Re: {Re} --')

        self.X_WS = self.X_WS / L
        self.Y_WS = self.Y_WS / L
        self.T_WS = self.T_WS * W / L
        self.P_WS = (self.P_WS - P0) / rho / (W**2)
        self.U_WS = self.U_WS / W
        self.V_WS = self.V_WS / W

        self.X_PINN = X_PINN_full / L
        self.Y_PINN = Y_PINN_full / L
        self.T_PINN = T_PINN_full * W / L

        self.params['L'] = L
        self.params['W'] = W
        self.params['P0'] = P0
        self.params['rho'] = rho
        self.params['Re'] = Re
        self.params['dim_T_PINN'] = dim_T_PINN
        self.params['R'] = R

    def _split_validation_train(self,
                                WS_val_idx: np.ndarray = np.sort(np.random.choice(21, 13, replace=False))) -> None:
        # Datos de validación
        print("The station for validation are: ", end="")
        for idx in WS_val_idx[:-1]: print(idx, end=", ")
        print(f"{WS_val_idx[-1]}.")

        self.val_data = {
            'T': self.T_WS[WS_val_idx, :],
            'X': self.X_WS[WS_val_idx, :],
            'Y': self.Y_WS[WS_val_idx, :],
            'U': self.U_WS[WS_val_idx, :],
            'V': self.V_WS[WS_val_idx, :],
            'P': self.P_WS[WS_val_idx, :]
        }

        # Datos de entrenamiento (eliminar filas de validación)
        self.train_data = {
            'T': np.delete(self.T_WS, WS_val_idx, 0),
            'X': np.delete(self.X_WS, WS_val_idx, 0),
            'Y': np.delete(self.Y_WS, WS_val_idx, 0),
            'U': np.delete(self.U_WS, WS_val_idx, 0),
            'V': np.delete(self.V_WS, WS_val_idx, 0),
            'P': np.delete(self.P_WS, WS_val_idx, 0)
        }

        self.pinn_grid = {
            'T': self.T_PINN,
            'X': self.X_PINN,
            'Y': self.Y_PINN
        }

        self.params["WS_val_idx"] = WS_val_idx

    def plot_stations(self) -> None:
        if self._state_data_process:
            self.X_WS
            self.val_data

        else:
            raise ValueError("No se a ejecutado .process_data()")

        ...

    def return_data(self):
        if self._state_data_process:
            return self.train_data, self.val_data, self.pinn_grid, self.params
        else:
            raise ValueError("No se a ejecutado .process_data()")

class ProcessDataColombia:
    def __init__(self, filepath:Path):
        # Ruta del archivo
        self.filepath = filepath

        # Tiempos registados
        self._fecha_creacion = datetime.now()
        self._tiempo_de_carga: Optional[Any] = None

        # Verificación de la data procesada
        self._state_data_process = False

        # dictionary to record the parameneters
        self.params = dict()

    def load_data(self) -> None:
        '''
        Se genera la carga de los datos
        '''
        if not self.filepath.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {self.filepath}")
        print(f"Iniciando carga de {self.filepath}")
        t0 = datetime.now()
        wsdata = pd.read_parquet(self.filepath, engine="fastparquet")
        self._tiempo_de_carga = datetime.now() - t0
        # print(f"Documento cargado en {self._tiempo_de_carga:.2e} s.")
        print(f"Documento cargado en {self._tiempo_de_carga} s.")
        self.codigo_estacion = wsdata["codigo_estacion"]
        self.latitud = wsdata["latitud"]
        self.longitud = wsdata["longitud"]
        self.altura = wsdata["altura"]
        self.segundos = wsdata["segundos"]
        self.presion = wsdata["presion"]
        self.vel_u = wsdata["vel u"]
        self.vel_v = wsdata["vel v"]
        self.temperatura = wsdata["temperatura"]

    def process_data(self, **kwargs) -> None:

        print("--- Coordenadas Cartesianas y Proyecciones ---")
        self._process_coordinates_and_projections()
        print("--- OK ---")

        print("--- Eliminar NaNs ---")
        delete_kwargs = {k: v for k, v in kwargs.items() if k in ['num_stations']}
        self._delete_nans(**delete_kwargs)
        print("--- OK ---")

        print("--- Selección de días y ordenamiento por coor ---")
        # Filtrar kwargs: solo pasar los que acepta _filter_by_time_and_sort_by_coor
        filter_kwargs = {k: v for k, v in kwargs.items() if k in ['n_days', 'interval']}
        self._filter_by_time_and_sort_by_coor(**filter_kwargs)
        print("--- OK ---")

        print("--- Centrado y creación de la malla ---")
        centered_kwargs = {k: v for k, v in kwargs.items() if k in ['R', 'rho', 'nu']}
        self._centered_grid_adimensionalization(**centered_kwargs)
        print("--- OK ---")

        print("--- Separar entre validación y entrenamiento ---")
        # Filtrar kwargs: solo pasar los que acepta _split_validation_train
        split_kwargs = {k: v for k, v in kwargs.items() if k in ['WS_val_idx']}
        self._split_validation_train(**split_kwargs)
        print("--- OK ---")

        self._state_data_process = True

    def _process_coordinates_and_projections(self) -> None:
        # Coordenadas Cartesianas y Proyecciones
        self.T_WS = self.segundos
        self.X_WS = np.array(6378000 * np.sin(np.radians(self.longitud)))
        self.Y_WS = np.array(6378000 * np.sin(np.radians(self.latitud)))
        self.Z_WS = np.array(self.altura)
        self.Temp_WS = np.array(self.temperatura)
        self.U_WS = self.vel_u
        self.V_WS = self.vel_v
        self.P_WS = self.presion * 100
        self.P_WS = self.P_WS * (1 - 0.0065 * self.Z_WS / (self.Temp_WS + 273.15 + 0.0065 * self.Z_WS))**(-5.257)

    def _delete_nans(self, num_stations:int = 7) -> None:
        T_nan_index = np.argwhere(pd.isna(self.segundos))
        # Eliminar NaNs de campos temporales
        for arr in [self.X_WS, self.Y_WS, self.Z_WS, self.U_WS,
                    self.V_WS, self.P_WS, self.Temp_WS]:
            arr = np.delete(arr, T_nan_index[:, 0], 0) # type: ignore

        self.T_WS = self._reshape_data(self.T_WS, num_stations)
        self.X_WS = self._reshape_data(self.X_WS, num_stations)
        self.Y_WS = self._reshape_data(self.Y_WS, num_stations)
        self.Z_WS = self._reshape_data(self.Z_WS, num_stations)
        self.U_WS = self._reshape_data(self.U_WS, num_stations)
        self.V_WS = self._reshape_data(self.V_WS, num_stations)
        self.P_WS = self._reshape_data(self.P_WS, num_stations)
        self.Temp_WS = self._reshape_data(self.Temp_WS, num_stations)

        # Eliminar estaciones con NaNs en ubicación
        X_nan_index = np.argwhere(np.isnan(self.X_WS))
        # Nota: Se asume eliminación simétrica en todas las variables
        self.T_WS = np.delete(self.T_WS, X_nan_index[:, 0], 0)
        self.P_WS = np.delete(self.P_WS, X_nan_index[:, 0], 0)
        self.U_WS = np.delete(self.U_WS, X_nan_index[:, 0], 0)
        self.V_WS = np.delete(self.V_WS, X_nan_index[:, 0], 0)
        self.X_WS = np.delete(self.X_WS, X_nan_index[:, 0], 0)
        self.Y_WS = np.delete(self.Y_WS, X_nan_index[:, 0], 0)
        self.Z_WS = np.delete(self.Z_WS, X_nan_index[:, 0], 0)
        self.Temp_WS = np.delete(self.Temp_WS, X_nan_index[:, 0], 0)

    def _reshape_data(self, arr: np.ndarray, num_stations: int) -> None:
        # Reestructurar matrices: 7 estaciones x mediciones
        return np.reshape(arr, (int(arr.shape[0] / num_stations), num_stations), order="F").T

    def _filter_by_time_and_sort_by_coor(self, n_days: int = 31, interval: int = 1) -> None:
        """
        interval is the value to indicate the time difference between records.
        1 indicate 1 h
        2 indicate 2 h
        .
        .
        .
        24 indicate 24 h or 1 day
        """
        # Selection of days
        one_day = 24 * 3600
        seconds_days = one_day * n_days
        seconds_days_args = self.T_WS[0,:] <= seconds_days

        self.T_WS = self.T_WS[:,seconds_days_args]
        self.X_WS = self.X_WS[:,seconds_days_args]
        self.Y_WS = self.Y_WS[:,seconds_days_args]
        self.Z_WS = self.Z_WS[:,seconds_days_args]
        self.U_WS = self.U_WS[:,seconds_days_args]
        self.V_WS = self.V_WS[:,seconds_days_args]
        self.P_WS = self.P_WS[:,seconds_days_args]
        self.Temp_WS = self.Temp_WS[:,seconds_days_args]

        self.params["n_days"] = n_days
        print(f"---> {n_days} selected days")

        # Selection of interval
        self.T_WS = self.T_WS[:, ::interval]
        self.X_WS = self.X_WS[:, ::interval]
        self.Y_WS = self.Y_WS[:, ::interval]
        self.Z_WS = self.Z_WS[:, ::interval]
        self.U_WS = self.U_WS[:, ::interval]
        self.V_WS = self.V_WS[:, ::interval]
        self.P_WS = self.P_WS[:, ::interval]
        self.Temp_WS = self.Temp_WS[:, ::interval]

        self.params["interval"] = interval
        print(f"---> Interval of {self.T_WS[0,1] / 60} min")


        # Order by coor X
        for snap in range(self.T_WS.shape[1]):
            idx_sort = np.argsort(self.X_WS[:, snap])
            self.T_WS[:, snap] = self.T_WS[idx_sort, snap]
            self.X_WS[:, snap] = self.X_WS[idx_sort, snap]
            self.Y_WS[:, snap] = self.Y_WS[idx_sort, snap]
            self.Z_WS[:, snap] = self.Z_WS[idx_sort, snap]
            self.U_WS[:, snap] = self.U_WS[idx_sort, snap]
            self.V_WS[:, snap] = self.V_WS[idx_sort, snap]
            self.P_WS[:, snap] = self.P_WS[idx_sort, snap]
            self.Temp_WS[:, snap] = self.Temp_WS[idx_sort, snap]

    def _centered_grid_adimensionalization(self, R:float = 0.2, rho:float = 1.269, nu:float = 1.382e-5) -> None:
        # Centrado
        x_min, x_max = np.min(self.X_WS), np.max(self.X_WS)
        y_min, y_max = np.min(self.Y_WS), np.max(self.Y_WS)
        t_min, t_max = np.min(self.T_WS), np.max(self.T_WS)

        self.X_WS = self.X_WS - (x_min + x_max) / 2
        self.Y_WS = self.Y_WS - (y_min + y_max) / 2
        self.T_WS = self.T_WS - t_min
        # self.T_WS = (self.T_WS - t_min) / (t_max - t_min)

        # Grilla PINN
        T_PINN = self.T_WS[0:1, :]
        R_PINN = 6378000 * np.sin(np.radians(R))
        x_PINN = np.arange(x_min - R_PINN, x_max + R_PINN, R_PINN) - (x_min + x_max) / 2
        y_PINN = np.arange(y_min - R_PINN, y_max + R_PINN, R_PINN) - (y_min + y_max) / 2

        X_PINN_grid, Y_PINN_grid = np.meshgrid(x_PINN, y_PINN)
        X_PINN_flat = X_PINN_grid.flatten('F')[:, None]
        Y_PINN_flat = Y_PINN_grid.flatten('F')[:, None]

        dim_T_PINN = T_PINN.shape[1]
        dim_N_PINN = X_PINN_flat.shape[0]

        T_PINN_full = np.tile(T_PINN, (dim_N_PINN, 1))
        X_PINN_full = np.tile(X_PINN_flat, dim_T_PINN)
        Y_PINN_full = np.tile(Y_PINN_flat, dim_T_PINN)

        # Adimensionalización
        p_min , p_max = np.nanmin(self.P_WS) , np.nanmax(self.P_WS)
        L = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        W = np.sqrt(np.nanmax(np.abs(self.U_WS))**2 + np.nanmax(np.abs(self.V_WS))**2)
        Re = int(W * L / nu)
        P0 = np.nanmean(self.P_WS) / (p_max - p_min)

        print(f'---> L: {L:.2f}, W: {W:.2f}, P0: {P0:.2f}, Re: {Re} --')

        self.X_WS = self.X_WS / L
        self.Y_WS = self.Y_WS / L
        self.T_WS = self.T_WS * W / (L * 3)
        self.P_WS = (self.P_WS - p_min) / (p_max - p_min)
        self.P_WS = (self.P_WS - P0)
        self.U_WS = self.U_WS / W
        self.V_WS = self.V_WS / W

        self.X_PINN = X_PINN_full / L
        self.Y_PINN = Y_PINN_full / L
        self.T_PINN = T_PINN_full * W / (L * 3)

        self.params['L'] = L
        self.params['W'] = W
        self.params['P0'] = P0
        self.params['rho'] = rho
        self.params['Re'] = Re
        self.params['dim_T_PINN'] = dim_T_PINN
        self.params['R'] = R

        print(f"{self.X_PINN.shape=} , {self.Y_PINN.shape=}")
        print(f"{np.nanmin(self.T_WS)=} , {np.nanmax(self.T_WS)=}")
        print(f"{np.nanmin(self.P_WS)=} , {np.nanmax(self.P_WS)=}")
        print(f"{np.nanmin(self.U_WS)=} , {np.nanmax(self.U_WS)=}")
        print(f"{np.nanmin(self.V_WS)=} , {np.nanmax(self.V_WS)=}")
        print(f"{np.nanmin(self.X_WS)=} , {np.nanmax(self.X_WS)=}")
        print(f"{np.nanmin(self.Y_WS)=} , {np.nanmax(self.Y_WS)=}")

    def _split_validation_train(self,
                                WS_val_idx: np.ndarray = np.sort(np.random.choice(7, 3, replace=False))) -> None:
        # Datos de validación
        print("The station for validation are: ", end="")
        for idx in WS_val_idx[:-1]: print(idx, end=", ")
        print(f"{WS_val_idx[-1]}.")

        self.val_data = {
            'T': self.T_WS[WS_val_idx, :],
            'X': self.X_WS[WS_val_idx, :],
            'Y': self.Y_WS[WS_val_idx, :],
            'U': self.U_WS[WS_val_idx, :],
            'V': self.V_WS[WS_val_idx, :],
            'P': self.P_WS[WS_val_idx, :]
        }

        # Datos de entrenamiento (eliminar filas de validación)
        self.train_data = {
            'T': np.delete(self.T_WS, WS_val_idx, 0),
            'X': np.delete(self.X_WS, WS_val_idx, 0),
            'Y': np.delete(self.Y_WS, WS_val_idx, 0),
            'U': np.delete(self.U_WS, WS_val_idx, 0),
            'V': np.delete(self.V_WS, WS_val_idx, 0),
            'P': np.delete(self.P_WS, WS_val_idx, 0)
        }

        self.pinn_grid = {
            'T': self.T_PINN,
            'X': self.X_PINN,
            'Y': self.Y_PINN
        }

        self.params["WS_val_idx"] = WS_val_idx


    def return_data(self):
        if self._state_data_process:
            return self.train_data, self.val_data, self.pinn_grid, self.params
        else:
            raise ValueError("No se a ejecutado .process_data()")

    def plot_estacion(self):
        x_plot = self.X_WS[:,0]
        y_plot = self.Y_WS[:,0]
        target = range(self.X_WS.shape[0])
        plt.figure(figsize=(8, 5))
        plt.scatter(x_plot, y_plot)
        # Añadir etiquetas con los números de estación
        for i, (x, y) in enumerate(zip(x_plot, y_plot)):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Ubicación de Estaciones')
        plt.grid(True)
        plt.show()

    def export_data(self, data_file_path: Path):
        np.savez(data_file_path,
                    T_WS = self.T_WS,
                    X_WS = self.X_WS,
                    Y_WS = self.Y_WS,
                    U_WS = self.U_WS,
                    V_WS = self.V_WS,
                    P_WS = self.P_WS,
                    T_PINN = self.T_PINN,
                    X_PINN = self.X_PINN,
                    Y_PINN = self.Y_PINN,
                    )
        print(f"Data guardada en: {data_file_path}")


###########################

