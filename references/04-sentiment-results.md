# Análisis de Features y Sentimiento — Resumen Ejecutivo

## Estado actual del proyecto

### Features totales: 21 por ticker
| Categoría | Cantidad | Descripción |
|-----------|----------|-------------|
| **Técnicas base** | 18 | Retorno, volatilidad, RSI, momentum, medias móviles, rangos, lags, calendario |
| **Sentimiento** | 3 | score, media móvil 5d, volatilidad 7d |

### Modelos entrenados: Random Forest con Optuna

| Ticker | Accuracy (baseline) | Accuracy (+sent) | Δ Accuracy | AUC (baseline) | AUC (+sent) | Δ AUC |
|--------|---------------------|-------------------|------------|----------------|-------------|-------|
| AAPL | 0.660 | **0.661** | ✓+0.11pp | 0.539 | **0.566** | ✓+2.70pp |
| MSFT | 0.448 | **0.528** | ✓+8.00pp | 0.530 | **0.541** | ✓+1.17pp |
| NVDA | 0.482 | **0.499** | ✓+1.69pp | 0.473 | **0.497** | ✓+2.37pp |
| AMZN | 0.608 | **0.622** | ✓+1.35pp | 0.574 | 0.572 | -0.23pp |
| TSLA | 0.586 | **0.597** | ✓+1.12pp | 0.570 | **0.572** | ✓+0.21pp |
| GOOGL | 0.523 | 0.512 | -1.01pp | 0.531 | 0.531 | -0.05pp |
| META | 0.567 | 0.554 | -1.30pp | 0.536 | 0.532 | -0.39pp |

### Conclusiones

1. **Sentimiento sintético funciona**: 5/7 tickers mejoraron accuracy, 4/7 mejoraron AUC
2. **Mejora promedio**: +1.42pp accuracy, +0.83pp AUC
3. **Mejor ticker**: AAPL con 66.1% accuracy y 0.566 AUC
4. **Problema conocido**: GOOGL y META empeoraron ligeramente (sentimiento puede ser ruido)

### Limitaciones conocidas

1. **GDELT bloqueado**: Rate limits estrictos impiden descargar noticias históricas
2. **Sentimiento sintético**: Basado en retornos pasados (proxy, no noticias reales)
3. **Cobertura real**: Solo ~100 noticias RSS por ticker (jul-ago 2026)

### Soluciones propuestas

| Prioridad | Solución | Esfuerzo |
|-----------|----------|----------|
| **Alta** | NewsAPI con API key (200 req/día gratis) | 1 día |
| **Media** | Financial PhraseBank (dataset pre-compilado) | 0.5 días |
| **Baja** | Esperar 24h y reintentar GDELT | 0 días |

### Archivos generados

```
data/interim/features_*_ml_ready.csv    # 21 features por ticker
reports/comparacion_sentimiento.csv     # Comparativa baseline vs sentimiento
reports/resultados_*.csv                # Métricas por ticker
models/rf_*.pkl                         # Modelos entrenados
```

### Próximos pasos

1. **Resolver sentimiento real**: NewsAPI o Financial PhraseBank
2. **Añadir features de tasa de cambio**: `volume_change`, `price_momentum_change`
3. **Cross-validation temporal**: Walk-forward analysis
4. **Ensemble**: Combinar modelos por ticker
5. **Backtesting**: Simular estrategia de trading
