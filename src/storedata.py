from abc import ABC, abstractmethod
from logging import getLogger
from pymongo import MongoClient
from typing import List, Dict

logger = getLogger(__name__)

class DataStorer(ABC):

    @abstractmethod
    def store(self):
        """Almacena los datos"""
        pass

class MongoStorer(DataStorer):
    def __init__(self, uri:str, db:str, collection:str):
        super().__init__()
        self.uri = uri
        self.db = db
        self.collection = collection
        self._total_stored_docs = 0
        logger.debug("MongoStorer OK")

    def store(self, docs: List[Dict]):

        try:
            # Conexión a la base de datos
            client = MongoClient(self.uri)
            db = client[self.db]
            collection = db[self.collection]
            # Carga a la base de datos
            collection.insert_many(docs)
            self._total_stored_docs += len(docs)
            logger.info(f"Insertados {self._total_stored_docs} docs.")

        except Exception as e:
            logger.error(f"Se presenta error cargando docs a la MongoDB: {e}")
            raise
