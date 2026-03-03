import logging
import pandas as pd

from typing import Protocol, Any

logger = logging.getLogger(__name__)


class DataBaseRepository(Protocol):
    """
    Contrato universal para la persistencia de los datos.
    Define las operaciones necesarias para la ingesta
    y filtrado de los datos
    """

    def connect(self) -> bool:
        """Establece conexión con la base de datos"""
        pass

    def insert_records(self, data:dict) -> None:
        """Inserta registros de forma masiva"""
        pass

    def get_filtered_collection(self, filters: dict[str, Any]) -> pd.DataFrame:
        """Retorna una coleción de datos filtrados"""
        pass