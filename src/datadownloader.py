import sys
import pyodbc
import logging
import requests
import pandas as pd

from io import StringIO
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Generator, List, Dict


logger = logging.getLogger(__name__)

class DataLoader(ABC):
    @abstractmethod
    def fetch_all_chunks(self):
        """Retoma un generador de DataFrames"""
        pass

class DataSaver(ABC):
    @abstractmethod
    def save(self, df: pd.Dataframe):
        """Guarda una porción de datos"""
        pass

class ManagerDownSaveData(ABC):

    def __init__(self, loader: DataLoader, saver: DataSaver):
        self.loader = loader
        self.saver = saver

    @abstractmethod
    def run(self):
        for chunk_df in self.loader.fetch_all_chunks():
            if not chunk_df.empty():
                self.saver.save(chunk_df)
                logger.info(f"Procesados {len(chunk_df)} registros.")
        ...

class SQLServerDataSaver(DataSaver):
    def __init__(self, database:str, query:str, server:str = "localhost\\SQLEXPRESS"):
        self.database = database
        self.query = query
        self.server = server
        self.conn_str = (
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'Trusted_Connection=yes;'
        )

    def save(self, df:pd.DataFrame) -> None:
        """
        Almacenar DataFrames de pandas en una base de datos de SQL Server
        """
        try:
            with pyodbc.connect(self.conn_str) as conn:
                cursor = conn.cursor()
                cursor.fast_executemany = True
                logger.info(f"Guardando {len(df)} registros.")
                try:
                    records = df.values.tolist()
                    cursor.executemany(self.query, records)
                    conn.commit()
                except Exception as e_db:
                    logger.error(f"Error insertando bloque: {e_db}.")
                    conn.rollback()

            ...
        except pyodbc.Error as e_conn:
            logger.error(f"Error de conexión a la base de datos: {e_conn}")
            ...



class APIDataDownloader(DataLoader):
    def __init__(self, url:str, token:str, soql_base_query:str, limit:int=1000, offset:int=0, max_retries:int = 3):
        self.url = url
        self.token = token
        self.soql_base_query = soql_base_query
        self.limit = limit
        self.max_retries = max_retries
        self.headers = {
            "X-App-Token": token,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", # Evita el bloqueo 403 por User-Agent
        }

    def fetch_all_chunks(self):
        offset = 0
        while True:
            soql_query = f"{self.soql_base_query} LIMIT {self.limit} OFFSET {offset}"
            try:
                response = self._request_with_retries()
                if response is not None:
                    df_chunk = pd.read_csv(StringIO(response.text), dtype=str)
                    if df_chunk.empty:
                        logger.info("Descarga finalizada: No hay más datos para consultar.")
                        break
                    
                    else:
                        # Seleccionar solo las 12 columnas necesarias en el orden correcto
                        columnas_necesarias = [
                            'codigoestacion', 'codigosensor', 'fechaobservacion', 'valorobservado',
                            'nombreestacion', 'departamento', 'municipio', 'zonahidrografica',
                            'latitud', 'longitud', 'descripcionsensor', 'unidadmedida'                    
                        ]
                        df_chunk = df_chunk[columnas_necesarias]

                        # Convertir tipos de datos
                        df_chunk["valorobservado"] = pd.to_numeric(df_chunk["valorobservado"], errors='coerce')
                        df_chunk["latitud"] = pd.to_numeric(df_chunk["latitud"], errors='coerce')
                        df_chunk["longitud"] = pd.to_numeric(df_chunk["longitud"], errors='coerce')
                        df_chunk["fechaobservacion"] = pd.to_datetime(df_chunk["fechaobservacion"], errors='coerce')

                        # Reemplazar NaN con None para que pyodbc lo maneje correctamente
                        df_chunk = df_chunk.where(pd.notna(df_chunk), None)
                        yield df_chunk
                        offset += self.limit
                else:
                    raise ValueError(f"{response=}")

            except Exception as e:
                logger.error(f"Se presenta el siguiente error: {e}")
            ...

    def _request_with_retries(self):
        soql_query = f"{self.soql_base_query} LIMIT {self.limit} OFFSET {offset}"
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.url, headers=self.headers, params={"query":soql_query})
                if response.status_code == 200:
                    return response
                else:
                    logger.warning(f"Error {response.status_code}: {response.text}")
                    return None
                ...
            except Exception as e:
                logger.error(f"Ocurrio un error realizando la solicitud: {e}")
                raise
        ...

def main():
    pass

if __name__=="__main__":
    sys.path.append("..")
    from config.logging_config import setup_imperative_production_config
    from logging import getLogger
    setup_imperative_production_config()
    logger = getLogger(__name__)
    main()