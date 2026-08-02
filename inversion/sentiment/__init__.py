"""inversion.sentiment — Sentimiento financiero por ticker.

- fetch.py   (SENT-001): descarga noticias RSS y deja data/raw/news_<TICKER>.csv
- analyzer.py (SENT-002): puntúa titulares con VADER + léxico financiero y agrega
  el score compuesto por día y ticker.
"""
