import logging
import logging.config
import sys
from typing import Optional, Callable
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)
yaml_file_path = Path('logging_config.yaml')

def setup_global_config() -> None:
    """
    Configuración del formato global
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s] [%(name)s] - %(message)s',
        stream=sys.stdout
    )
    
def setup_imperative_production_config(log_file_name: str = "train_pinn.log") -> None:
    """
    log_filename: Nombre del archivo donde se guardara los registros.
    """
    
    # Se definen los formatos
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] (%(lineno)d) %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        fmt='[%(asctime)s] (%(levelname)s) %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Se crean los manejadores (Handlers)
    # Consola (INFO +) Solo información relevante
    consola_handler = logging.StreamHandler(sys.stdout)
    consola_handler.setLevel(logging.INFO)
    consola_handler.setFormatter(simple_formatter)
    
    # Archivo (DEBUG +) Registro forense completo
    file_handler = logging.FileHandler(log_file_name, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Configurar el Logger Raiz (Root Logger)
    root_logger = logging.getLogger()   # La raiz de la jerarquia
    root_logger.setLevel(logging.DEBUG) # Nivel mínimo global
    
    # Limpieza de handlers existentes para evitar duplicados
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    # Añadir los nuevos handlers configurados
    root_logger.addHandler(consola_handler)
    root_logger.addHandler(file_handler)
    
    # # Configuración especifica para módulos
    # model_logger = logging.getLogger('model.name')
    # model_logger.setLevel(logging.WARNING) # Más silencioso

def setup_production_config(
        log_file_name: Optional[str] = None,
        yaml_file_path:str = 'logging_config.yaml', 
        setup_imperative:Callable = setup_imperative_production_config,
        ) -> None:
    """
    Carga la configuración desde YAML,
    en caso contrario, carga la 
    configuración almacenda.
    """
    try:
        if not isinstance(yaml_file_path, Path):
            yaml_file_path = Path(yaml_file_path)

        if yaml_file_path and yaml_file_path.exists():
            with open(yaml_file_path, 'r') as yaml_file:
                config_log = yaml.safe_load(yaml_file)

            if log_file_name is not None:
                config_log["handlers"]["file"]["filename"] = log_file_name

            logging.config.dictConfig(config_log)
            logger.info(f"Configuración cargada desde {yaml_file_path.name}")
        else:
            raise FileNotFoundError(f"No se encontro el archivo {yaml_file_path}")
        pass

    except (FileNotFoundError, AttributeError) as e:
        logger.warning(f"{e}: Iniciando ejecución imperativa ...")
        if log_file_name is not None:
            setup_imperative(log_file_name)
        else:
            setup_imperative()


    except Exception as e:
        logger.error(f"Error inesperado al configurar el logging: {e}")
    pass
    
    