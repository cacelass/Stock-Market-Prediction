import pandas as pd
from inversion.utils import paths 
import yfinance as yf

def get_yfinance_history(ticker, start, end):
    """Descarga 20 años de historia diaria usando Yahoo Finance
     Retorna un DataFrame con columnas: 
     ['timestamp', 'open', 'high', 'low', 'close', 'volume']"""
    print(f"   ⬇ Descargando YFinance para {ticker}...")
    try:
        # Descarga
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        
        # Aplanar MultiIndex si existe (corrección común en versiones recientes de yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        
        # Renombrar columnas a minúsculas estándar
        df = df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        
        # Asegurar formato fecha
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        print(f"   ⚠ Error con YFinance: {e}")
        return pd.DataFrame()

def load_data(ticker=None):
    """Carga los datos crudos desde el archivo CSV.
     Usa la ruta definida en paths.py"""
    try:
        # Usa la variable RAW_DATA_FILE definida en paths.py
        df = pd.read_csv(paths.RAW_DATA_FILE, parse_dates=["timestamp"]) 
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en {paths.RAW_DATA_FILE}")
        return None
    except Exception as e:
        print(f"❌ Error inesperado cargando datos: {e}")
        return None