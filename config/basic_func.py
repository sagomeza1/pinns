import datetime

def now() -> str:
    """
    Devuelve la fecha y hora en siguiente formato:
    16 de agosto del 2024 a las 2.47pm = 240816_1447
    """
    t0 = datetime.datetime.now()
    return f"{str(t0.year)[-2:]}{t0.month:02}{t0.day:02}_{t0.hour:02}{t0.minute:02}"