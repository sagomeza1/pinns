import logging

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    """
    Pipe line para la ingesta de los datos:
    - Obtiene los datos para almacenarlos en una base de datos.
    - Filtra y genera una colección de datos
    - Los datos son almacenados.
    """
    def __init__(self, getter, db, storage):
        self.getter
        self.db
        self.storage
        pass

    def run(self):
        try:
            logger.info("Inicio Pipeline de ingesta.")
            for raw_data in self.getter.get_data():
                self.db.save(raw_data)
                pass
            logger.info("Datos almacenados.")
            processed = self.db.filter_and_join()
            logger.info("Colección de datos generada.")
            self.storage.save_parquet(processed)
            logger.info("Colección de datos almacenada.")
            logger.info("Finalizado Pipeline de ingesta")
        except Exception as e:
            logger.error(f"Error en el pipeline de ingesta: {e}")
            raise