import sys
from abc import ABC, abstractmethod

from pathlib import Path
from logging import getLogger

logger = getLogger(__name__)

class DataLoader:
    @abstractmethod
    def load_data(self):
        """Carga de la información"""
        pass

class DataProcessor(ABC):
    @abstractmethod
    def process_data(self):
        """Implementación del procesamiento de los datos"""
        pass