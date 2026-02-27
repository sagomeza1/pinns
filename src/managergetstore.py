from logging import getLogger
from abc import ABC, abstractmethod

logger = getLogger(__name__)

from .getdata import WSDataGetter
from .storedata import DataStorer

class GetStoreManager(ABC):

    def __init__(self, getter: WSDataGetter, storer: DataStorer) -> None:
        super().__init__()
        self.getter = getter
        self.storer = storer
        logger.debug("GetStoreManager OK")

    def run(self):
        """Ejecuta el proceso de carga y almacenamiento de los datos"""
        for chunk in self.getter.get_data():
            if len(chunk) != 0:
                self.storer.store(chunk)
                logger.debug(f"Procesados {len(chunk)} registros.")

def main():
    pass

if __name__=='__main__':
    main()