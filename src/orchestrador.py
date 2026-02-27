#%%
import logging

from pathlib import Path
from src.getdata import APIGetter
from src.storedata import MongoStorer
from src.managergetstore import GetStoreManager
from config.orchestrador_config import load_orchestrador_config
#%%
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s | %(name)s | %(lineno)d | %(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)
#%%
class Orchestrador:
    def __init__(self, config_path:str):
        self.config_path = config_path
        pass

    def api_mongo(self):
        getter_config, storer_config = load_orchestrador_config(self.config_path)

        for collection in storer_config.collections:
            logger.info(f"{collection=}")
            getter = APIGetter(
                url=getter_config.url_store[collection],
                soql_base_query=getter_config.soql_base_query,
                token=getter_config.token,
            )

            storer = MongoStorer(
                uri=storer_config.uri,
                db=storer_config.db,
                collection=collection
            )

            manager = GetStoreManager(
                getter=getter,
                storer=storer
            )

            manager.run()

            pass

        pass

    pass

def main():
    from pathlib import Path
    path = Path().cwd() / "config" / "data.yaml"
    logger.debug(f"{path.exists()=}")
    orchestrador = Orchestrador(path)
    orchestrador.api_mongo()

if __name__=='__main__':
    main()