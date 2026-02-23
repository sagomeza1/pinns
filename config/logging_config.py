import logging
import sys

def setup_global_config() -> None:
    """
    Configuración del formato global
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s] [%(name)s] - %(message)s',
        stream=sys.stdout
    )
    
def setup_imperative_production_config(log_filename: str = "train_pinn.log") -> None:
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
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
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
    
    