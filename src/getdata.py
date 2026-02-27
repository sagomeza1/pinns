from logging import getLogger
import requests
from abc import ABC, abstractmethod
from typing import Optional, Dict, Generator

logger = getLogger(__name__)

class WSDataGetter(ABC):
    """
    Clase para obtener datos de
    las estaciones meteorológicas
    """
    def __init__(self, tipo:Optional[str] = None):
        super().__init__()
        self.tipo: str = tipo if tipo is not None else 'NA'

    @abstractmethod
    def get_data(self):
        """Obtener datos"""
        pass

class APIGetter(WSDataGetter):
    """
    Clase para obtener datos de
    las estaciones meteorológicas
    desde una API
    """
    def __init__(self,
                 url:str,
                 soql_base_query:str,
                 token:Optional[str] = None,
                 limit:Optional[int] = None,
                 max_retries:Optional[int] = None,
                 tipo:str = 'API',
                 ):

        super().__init__(tipo)
        self.url = url
        self.soql_base_query = soql_base_query
        self.token = token
        self.limit = limit if limit is not None else 1000
        self.max_retries = max_retries if max_retries is not None else 3
        logger.debug("APIGetter OK")

    def get_data(self):
        """Función que itera la solicitud a la API"""
        offset = 0
        try:
            # Se define el headers de acuerdo a la disponibilidad del token
            if self.token is not None:
                headers = {
                    "X-App-Token": self.token,
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", # Evita el bloqueo 403 por User-Agent
                }
            else:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            # Se inicia la iteración
            while True:
                soql_query = f"""
                    {self.soql_base_query}
                    limit {self.limit}
                    offset {offset}
                    """
                # Se realiza la solicitud
                response = self._request_with_retries(
                    headers=headers,
                    params={'query': soql_query}
                )
                if response is not None:
                    if len(response.json()) == 0:
                        logger.info(f"Descarga finalizada: No hay más datos para consultar.")
                        break
                    else:
                        yield response.json()
                        offset += self.limit
                else:
                    raise ValueError(f"{response=}")

        except Exception as e:
            logger.error(f"Se presenta error cargando datos de la API: {e}.")

    def _request_with_retries(
            self,
            headers:Dict[str, str],
            params:Dict[str, str]
    ):
        for attemp in range(self.max_retries):
            try:
                # Se realiza la solicitud
                response = requests.get(
                    url=self.url,
                    params=params,
                    headers=headers
                )
                logger.debug(f"{response.status_code=}")
                if response.status_code == 200:
                    return response
                else:
                    logger.warning(f"Intento {attemp + 1} (Error {response.status_code}) {response.text}")
            except Exception as e:
                logger.error(f"Se presenta error realizando la solicitud a la API: {e}.")
                raise



