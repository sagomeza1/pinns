
import os
import yaml

from logging import getLogger
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

logger = getLogger(__name__)

load_dotenv()
uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
token = os.getenv('TOKEN', None)

@dataclass
class GetterConfig:
    url_store: Dict[str, str]
    soql_base_query: str
    token: Optional[str] = None
    limit: Optional[int] = None
    max_retries:Optional[int] = None

@dataclass
class StorerConfig:
    uri: str
    db: str
    collections: List[str]

def load_orchestrador_config(config_yaml_file: str) -> Tuple[
    GetterConfig,
    StorerConfig,
    ]:
    """Entrega la configuración para la orquestación"""

    with open(config_yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    getter_config = GetterConfig(
        url_store=config['urls'],
        token=token,
        **config['getter_config'],
    )

    storer_config = StorerConfig(
        uri=uri,
        **config['storer_config']
    )

    return getter_config, storer_config