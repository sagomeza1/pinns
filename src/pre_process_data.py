# %%
# Librerias
import urllib
import inspect
import requests
import warnings
import sqlalchemy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from scipy.interpolate import interp1d
from sqlalchemy import create_engine, Engine
from typing import Callable, List, Dict, Any, Optional

# %%
# Clase StationMetrics y FilterStrategy


@dataclass
class StationMetrics:
    """
    Data Class para almacenar las métricas calculadas de una estación.
    """
    station_id: str
    min_date: pd.Timestamp
    max_date: pd.Timestamp
    unique_diffs: List[Any]
    row_count: int

# Definición de un tipo para la función de filtrado
# Recibe StationMetrics y devuelve un booleano
FilterStrategy = Callable[[StationMetrics], bool]

# %%
# Clase DataPreprocessor
class DataPreprocessor:
    """ Descripción:
    Clase encargada de la extracción y preprocesamiento de datos desde SQL Server
    hacia estructuras de datos de Pandas, utilizando SQLAlchemy para garantizar
    compatibilidad y rendimiento. Se entrega un DataFrame de pandas con las
    siguientes columnas:
    - codigo_estacion
    - latitud
    - longitud
    - altura
    - segundos
    - presion
    - vel_u
    - vel_v
    - temperatura
    """

    def __init__(self, server:str, database:str, interpolate:bool = False, **kwargs):
        """
        Inicializa el motor de conexión de SQLAlchemy.

        Parámetros:
        - server: Dirección o nombre del servidor SQL.
        - database: Nombre de la base de datos objetivo.
        """
        # Filtramos los avisos de SQLAlchemy sobre versiones de servidor no reconocidas
        # para evitar ruido en la consola cuando la conexión es funcional.
        warnings.filterwarnings('ignore', category=sqlalchemy.exc.SAWarning)

        self.server = server
        self.database = database
        self.interpolate = interpolate
        self.engine: Optional[Engine] = self._create_sql_engine()
        self.full_stations_data: Optional[pd.DataFrame] = None
        self.coor_stations_data: Optional[pd.DataFrame] = None
        self.clean_full_stations: Optional[pd.DataFrame] = None
        self.inter_full_stations: Optional[pd.DataFrame] = None
        self.global_config = kwargs

    def _create_sql_engine(self) -> Engine:
        """
        Crea un objeto Engine de SQLAlchemy optimizado para SQL Server.

        Retorna:
            sqlalchemy.engine.Engine: Objeto de conexión configurado.
        """
        connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"Trusted_Connection=yes;"
        )
        # Es necesario codificar la cadena para que sea compatible con la URL de SQLAlchemy
        params = urllib.parse.quote_plus(connection_string)
        engine_url = f"mssql+pyodbc:///?odbc_connect={params}"

        return create_engine(engine_url)

    def _filter_kwargs_for_function(self, target_function: Callable, **kwargs: Any) -> Dict[str, Any]:
        """
        Filtra un diccionario de argumentos para que contenga solo los parámetros
        que la función de destino acepta en su firma.

        Args:
            target_function: La función o método hacia el cual se enviarán los argumentos.
            **kwargs: El set total de argumentos disponibles.

        Returns:
            Dict[str, Any]: Un subconjunto de kwargs válidos para la función.
        """
        signature = inspect.signature(target_function)
        valid_params = signature.parameters.keys()
        return {k: v for k, v in kwargs.items() if k in valid_params}

    def load_data(self) -> None:
        """
        Ejecuta las consultas SQL y carga los resultados en DataFrames de Pandas,
        gestionando los recursos de conexión de forma segura.
        """
        full_query = "SELECT * FROM dbo.estaciones_full;"
        coor_query = "SELECT * FROM dbo.coordenadas_estaciones;"

        print("=" * 50)
        print(f"{' Iniciando carga de datos desde SQL Server ':^50}")
        print(f"{'Carga completada ':.<45}", end="")

        try:
            # El uso de 'with' asegura que la conexión se cierre automáticamente
            if self.engine is None:
                raise ConnectionError(f"\nEl motor de base de datos no fue inicializado.")

            with self.engine.connect() as connection:
                self.full_stations_data = pd.read_sql(full_query, connection)
                self.coor_stations_data = pd.read_sql(coor_query, connection)

            print(f"{' OK':.>5}")

        except sqlalchemy.exc.SQLAlchemyError as error:
            print(f"\n[ERROR CRÍTICO]: Fallo en la conexión o consulta: {error}\n")
        except Exception as error:
            print(f"\n[ERROR INESPERADO]: {error}\n")
        print("=" * 50)

    def process_data(self) -> None:
        """
        Procesa el resultado de las consulta para su formato final
        """
        print("=" * 50)
        print(f"{'Iniciando Procesamiento de los datos':^50}")
        # Copia de los dfs originales
        full_stations = self.full_stations_data.copy()
        coor_stations = self.coor_stations_data.copy()

        try:
            # Limpieza de los datos
            print(f"{'Limpieza de los datos ':.<45}", end="")
            clean_params = self._filter_kwargs_for_function(self._clean_dataframe, **self.global_config)
            clean_full_stations = self._clean_dataframe(full_stations, **clean_params)
            print(f"{' OK':.>5}")

            # Adición de la altura
            print(f"{'Adición de la altura ':.<45}", end="")
            alt_params = self._filter_kwargs_for_function(self._get_station_alts, **self.global_config)
            clean_full_stations = self._get_station_alts(clean_full_stations, **alt_params)
            print(f"{' OK':.>5}")

            # Eliminación de duplicados
            print(f"{'Eliminación de duplicados ':.<45}", end="")
            clean_full_stations.drop_duplicates(inplace=True)
            clean_full_stations.reset_index(drop=True, inplace=True)
            print(f"{' OK':.>5}")

            # Segundos, componentes X y Y de velocidad
            print(f"{'Segundos, Velocidad U, Velocidad V ':.<45}", end="")
            clean_full_stations = self._time_to_seconds(clean_full_stations)
            clean_full_stations = self._vel_u_and_vel_v(clean_full_stations)
            print(f"{' OK':.>5}")

            # Filtro por condiciones
            print(f"{'Filtro de estaciones ':.<45}", end="")
            filter_params = self._filter_kwargs_for_function(self._filter_stations, **self.global_config)
            clean_full_stations = self._filter_stations(clean_full_stations, **filter_params)
            print(f"{' OK':.>5}")

            # Complerar nans
            print(f"{'Homogenizar registros ':.<45}", end="")
            clean_full_stations = self._complete_nans(clean_full_stations)
            print(f"{' OK':.>5}")

            # Interpolación
            if self.interpolate:
                print(f"{'Interpolación ':.<45}", end="")
                inter_params = self._filter_kwargs_for_function(self._filter_stations, **self.global_config)
                inter_full_stations = self._filter_stations(clean_full_stations, **inter_params)
                self.inter_full_stations = inter_full_stations
                print(f"{' OK':.>5}")

        except Exception as e:
            print(f"ERROR")
            print(f"Error: {e}")
            return None

        self.clean_full_stations = clean_full_stations
        print("=" * 50)

    def _clean_dataframe(self, df:pd.DataFrame, criteria: FilterStrategy) -> pd.DataFrame:
        """
        Se filtran las estaciones a partir del filtro indicado
        """
        valid_stations = list()
        df.drop("tx_id", axis=1, inplace=True)
        df = df.groupby(["codigo_estacion", "fecha_observacion"])[["presion", "velocidad", "direccion", "temperatura"]].mean().reset_index()
        station_ids = df["codigo_estacion"].unique()
        for idx, station_id in enumerate(station_ids, 1):
            try:
                metrics = self._calculate_metrics(df, station_id)
                if criteria(metrics):
                    valid_stations.append(station_id)

            except Exception as e:
                print(f"Error procesando la estación {station_id}: {e}")

        return df[df["codigo_estacion"].isin(valid_stations)]

    def _calculate_metrics(self, df:pd.DataFrame, station_id:str) -> StationMetrics:
        """
        Calcula las métricas necesarias para una estación específica.

        Args:
            station_id (str): El código identificador de la estación.

        Returns:
            (StationMetrics): Un objeto con los datos procesados.
        """

        station_data = df[df["codigo_estacion"] == station_id]
        return StationMetrics(
            station_id=station_id,
            min_date=station_data["fecha_observacion"].min(),
            max_date=station_data["fecha_observacion"].max(),
            unique_diffs=station_data["fecha_observacion"].sort_values().diff().dropna().unique().tolist(),
            row_count=len(station_data)
        )
        ...

        ...

    def _get_station_alts(self, df:pd.DataFrame) -> pd.DataFrame:
        points = list()
        station_alts_dict = {"latitud":list(), "longitud":list(), "altura":list()}

        df["longitud"] = df["codigo_estacion"].map(dict(zip(self.coor_stations_data["codigo_estacion"], self.coor_stations_data["longitud"])))
        df["latitud"] = df["codigo_estacion"].map(dict(zip(self.coor_stations_data["codigo_estacion"], self.coor_stations_data["latitud"])))

        for idx in range(len(self.coor_stations_data)):
            points.append({"latitude": self.coor_stations_data.loc[idx, "latitud"], "longitude": self.coor_stations_data.loc[idx, "longitud"]})

        station_alts = self._get_point_alts(points)
        for station_alt in station_alts:
            station_alts_dict["latitud"].append(station_alt["latitude"])
            station_alts_dict["longitud"].append(station_alt["longitude"])
            station_alts_dict["altura"].append(station_alt["elevation"])

        coor_alt_df = pd.DataFrame(station_alts_dict)
        return pd.merge(df, coor_alt_df, "left", ["latitud", "longitud"])

    def _get_point_alts(self, points:List[Dict[str, float]]) -> Dict[str, float]:
        """
        Consulta la elevación de múltiples puntos usando la API de Open-Elevation.
        coordenadas: Lista de diccionarios [{'latitude': lat, 'longitude': lon}, ...]
        """
        url = "https://api.open-elevation.com/api/v1/lookup"
        payload = {"locations": points}

        response = requests.post(url, json=payload, timeout=20)
        if response.status_code == 200:
            return response.json()['results']
        return None
        ...

    def _time_to_seconds(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Los registros de fechas pasan a segundos para simular
        un tiempo continuo
        """
        init_date = df["fecha_observacion"].min()
        df["segundos"] = (df["fecha_observacion"] - init_date) / np.timedelta64(1, 's')
        df.drop(["fecha_observacion"], axis=1, inplace=True)
        return df

    def _vel_u_and_vel_v(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Se agregan los componentes de velocidad en los ejes X y Y.
        """
        df["vel_u"] = df.apply(lambda df: df["velocidad"] * np.sin(np.pi*(df["direccion"]/180)), axis=1)
        df["vel_v"] = df.apply(lambda df: df["velocidad"] * np.cos(np.pi*(df["direccion"]/180)), axis=1)
        df.drop(["velocidad", "direccion"], axis=1, inplace=True)
        return df
        ...

    def _filter_stations(self, df:pd.DataFrame, min_pressure:Optional[int] = None, max_altitude: Optional[int] = None, seconds: int = 3600) -> pd.DataFrame:
        if min_pressure:
            df = df[df["presion"] > min_pressure]
        if max_altitude:
            df = df[df["altura"] < max_altitude]
        
        df = df[df["segundos"] % seconds == 0]
        
        return df
        ...

    def _interpolate_data(self, df:pd.DataFrame, minutes_interval:int = 20) -> pd.DataFrame:
        seconds_interval = 60 * minutes_interval
        high_seconds = df["segundos"].max()
        interpolated_seconds = np.arange(0, high_seconds + seconds_interval, seconds_interval)

        interpolated_data_dict = dict()
        interpolated_data_dict["codigo_estacion"] = np.array([])
        interpolated_data_dict["segundos"] = np.array([])
        interpolated_data_dict["presion"] = np.array([])
        interpolated_data_dict["vel_u"] = np.array([])
        interpolated_data_dict["vel_v"] = np.array([])
        interpolated_data_dict["temperatura"] = np.array([])
        interpolated_data_dict["latitud"] = np.array([])
        interpolated_data_dict["longitud"] = np.array([])
        interpolated_data_dict["altura"] = np.array([])

        station_ids = df["codigo_estacion"].unique()
        for station_id in station_ids:
            seconds = df.loc[df["codigo_estacion"] == station_id, "segundos"]
            pressure = df.loc[df["codigo_estacion"] == station_id, "presion"]
            vel_u = df.loc[df["codigo_estacion"] == station_id, "vel_u"]
            vel_v = df.loc[df["codigo_estacion"] == station_id, "vel_v"]
            temperature = df.loc[df["codigo_estacion"] == station_id, "temperatura"]

            latitude = df.loc[df["codigo_estacion"] == station_id, "latitud"]
            longitude = df.loc[df["codigo_estacion"] == station_id, "longitud"]
            altitude = df.loc[df["codigo_estacion"] == station_id, "altura"]

            latitude = np.tile(latitude.values[0], len(interpolated_seconds))
            longitude = np.tile(longitude.values[0], len(interpolated_seconds))
            altitude = np.tile(altitude.values[0], len(interpolated_seconds))
            station_id = np.tile(station_id, len(interpolated_seconds))

            p_cubic = interp1d(seconds, pressure, kind="cubic")
            u_cubic = interp1d(seconds, vel_u, kind="cubic")
            v_cubic = interp1d(seconds, vel_v, kind="cubic")
            t_cubic = interp1d(seconds, temperature, kind="cubic")

            interpoled_p = p_cubic(interpolated_seconds)
            interpoled_u = u_cubic(interpolated_seconds)
            interpoled_v = v_cubic(interpolated_seconds)
            interpoled_t = t_cubic(interpolated_seconds)

            interpolated_data_dict["segundos"] = np.concat([interpolated_data_dict["segundos"], interpolated_seconds])
            interpolated_data_dict["presion"] = np.concat([interpolated_data_dict["presion"], interpoled_p])
            interpolated_data_dict["vel_u"] = np.concat([interpolated_data_dict["vel_u"], interpoled_u])
            interpolated_data_dict["vel_v"] = np.concat([interpolated_data_dict["vel_v"], interpoled_v])
            interpolated_data_dict["temperatura"] = np.concat([interpolated_data_dict["temperatura"], interpoled_t])
            interpolated_data_dict["latitud"] = np.concat([interpolated_data_dict["latitud"], latitude])
            interpolated_data_dict["longitud"] = np.concat([interpolated_data_dict["longitud"], longitude])
            interpolated_data_dict["altura"] = np.concat([interpolated_data_dict["altura"], altitude])
            interpolated_data_dict["codigo_estacion"] = np.concat([interpolated_data_dict["codigo_estacion"], station_id])
            
        return pd.DataFrame(interpolated_data_dict)

    def _complete_nans(self, df:pd.DataFrame) -> pd.DataFrame:
        segundos = df["segundos"].unique()
        estaciones = df["codigo_estacion"].unique()
        coor = df[["codigo_estacion", "latitud", "longitud"]].drop_duplicates()
        
        segundos , estaciones = np.meshgrid(segundos, estaciones)
        
        df_temp = pd.DataFrame({"codigo_estacion": estaciones.flatten(), "segundos": segundos.flatten()})
        df_temp["latitud"] = df_temp["codigo_estacion"].map(dict(zip(coor["codigo_estacion"], coor["latitud"])))
        df_temp["longitud"] = df_temp["codigo_estacion"].map(dict(zip(coor["codigo_estacion"], coor["longitud"])))
        
        return pd.merge(df_temp, df, "left", ["codigo_estacion", "segundos"])

    def export_data(self, save_file_path:Path, **kwargs) -> None:
        print("=" * 50)
        print(f"{'Iniciando Procesamiento de los datos':^50}")
        try:
            print(f"{'Exportando registro de las estaciones ':.<45}", end="")
            self.clean_full_stations.to_parquet(save_file_path, engine="fastparquet")
            print(f"{' OK':.>5}")
            
            if self.interpolate:
                print(f"{'Exportando registros interpolados ':.<45}", end="")
                save_interpolate_file_path = save_file_path.with_name(f"{save_file_path.stem}_interpolate{save_file_path.suffix}")
                self.inter_full_stations.to_parquet(save_interpolate_file_path, engine="fastparquet")
                print(f"{' OK':.>5}")
        except Exception as e:
            print("ERROR")
            print(f"Error: {e}")
            return None
        print("=" * 50)

    def time_diff(self) -> None:
        print("Diferencias registradas entre las fechas:")
        for diff in self.full_stations_data["fecha_observacion"].sort_values().diff().unique(): print(f"> {diff}")
        ...

    def info_stations(self) -> None: ...

# %%
def main():
    DIR_DATA_PATH = Path('c:/Users/User/Documents/pinns/data/raw')
    # Configuración de los parámetros de conexión
    def filter_by_min_rows(metrics: StationMetrics) -> bool:
        return metrics.row_count >= 700

    kwargs = {
        'server'        : 'localhost\\SQLEXPRESS',
        'database'      : 'EM_CAR',
        'interpolate'   : True,
        'criteria'      : filter_by_min_rows,
        'min_pressure'  : 800,
        'max_altitude'  : 1000,
        'save_file_path': DIR_DATA_PATH / "em_caribe_20251201_20251231.parquet"
    }
    preprocess = DataPreprocessor(**kwargs)
    preprocess.load_data()
    preprocess.process_data()
    preprocess.export_data(**kwargs)
    ...
if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProceso interrumpido manualmente")