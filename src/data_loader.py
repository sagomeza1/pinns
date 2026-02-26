import sys
from abc import ABC, abstractmethod

from pathlib import Path
from logging import getLogger
import numpy as np
from dataclasses import dataclass
from typing import Dict

logger = getLogger(__name__)

class DataLoader:
    @abstractmethod
    def load_data(self) -> Dict[str, np.ndarray]:
        """
        Carga de la información.
        Se entrega un diccionario con las
        siguientes llaves:
        dtm - fecha y hora
        lat - latitud
        lon - longitud
        alt - altura
        dir - dirección del viento
        vel - velocidad del viento
        pre - presión
        """
        pass

class DataProcessor(ABC):
    @abstractmethod
    def process_data(self):
        """Implementación del procesamiento de los datos"""
        pass