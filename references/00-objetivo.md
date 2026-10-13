# Objetivo del proyecto

## Pregunta a responder

Con datos de mercado OHLCV+volumen y sentimiento de noticias, se puede predecir el RIESGO (volatilidad realizada a 10 dias) de un activo mejor que la persistencia simple, y servir esa senal para dimensionar posicion? La direccion a 5 dias se mantiene como senal secundaria.

## Métrica de éxito

Modelo de volatilidad: R2 fuera de muestra >= 0.15 y Spearman >= 0.40 en walk-forward de los 7 tickers. Direccion (secundaria): AUC >= 0.55.

## Datos de partida

Precios diarios OHLCV de 7 tickers (20 anos, yfinance) con features tecnicas derivadas; sentimiento de noticias NewsAPI.ai. Sin datos intradia ni fundamentales.

## Criterio de parada

Si el modelo de volatilidad no bate la persistencia en R2 en al menos 5 de 7 tickers, o el backtest con costes no supera a buy&hold, el proyecto se replantea como analitico/educativo.

## Usuarios

Uso personal y de portfolio: gestion de riesgo y dimensionado de posicion. No es asesoramiento financiero.

## Alcance y no-alcance

ENTRA: forecasting de volatilidad/riesgo, intervalos de confianza, serving por API. NO ENTRA: ejecucion de ordenes reales, datos intradia, ni prediccion de direccion como objetivo principal.

## Restricciones y riesgos

Senales debiles en direccion (probado: AUC~0.55); pocos datos efectivos por ticker; no es asesoramiento financiero.

