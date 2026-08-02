# Papers de Referencia — Predicción de activos, sentimiento y ML

Generado por el agente `research` (OpenAlex) el 2026-08-02. Clasificado en
tres categorías que informan el diseño de features, modelos y sentimiento del
proyecto. Rankeado por relevancia.

## Sentimiento financiero

| Paper | Año | Por qué importa |
|-------|-----|-----------------|
| **Financial News Sentiment Analysis using Modified VADER for Stock Price Prediction** | 2022 | El mismo enfoque que este repo: VADER adaptado a jerga financiera para predecir precios. Valida la decisión de SENT-002 (léxico financiero embebido). |
| Measuring news sentiment | 2020 | Metodología para puntuar sentimiento de noticias de forma agregada; apoya el diseño de `sentiment_score`/`sentiment_ma5`. |
| A survey on sentiment analysis methods, applications, and challenges | 2022 | Panorama de métodos; útil para justificar por qué VADER (léxico, sin entrenamiento) vs. transformers. |
| Sentiment Analysis in the News | 2013 | Línea base histórica sobre agregación diaria de sentimiento de prensa. |
| Sentiment Analysis for Fake News Detection | 2021 | Advertencia sobre calidad del corpus (enlaza con el hallazgo de MODEL-002: corpus pequeño → señal débil). |

## Features técnicas

| Paper | Año | Por qué importa |
|-------|-----|-----------------|
| **Deep Learning for Stock Market Prediction** | 2020 | Revisa qué indicadores técnicos importan; alinea con `FEATURE_COLS` (RSI, MA, VWAP, volatilidad). |
| Prediction of stock price direction using a hybrid GA-XGBoost algorithm with a three-stage feature engineering | 2021 | Selección de features en 3 etapas; candidato a futuro (feature importance ya disponible en `ml`). |
| A graph-based CNN-LSTM stock price prediction algorithm with leading indicators | 2021 | Indicadores adelantados + grafos; línea de investigación si se activa graphify (TMPL-005). |
| Stock Market Analysis: A Review and Taxonomy of Prediction Techniques | 2019 | Taxonomía de técnicas; marco para documentar decisiones del repo. |

## Modelos ML

| Paper | Año | Por qué importa |
|-------|-----|-----------------|
| Stock Price Prediction Using Machine Learning and LSTM-Based Deep Learning Models | 2021 | Comparativa RF/LSTM; contextualiza por qué el baseline usa RandomForest (MODEL-001). |
| Stock price prediction using DEEP learning algorithm and its comparison with machine learning algorithms | 2019 | Evidencia de que modelos tabulares (RF/GBM) compiten con DL en series financieras. |
| Review of deep learning: concepts, CNN architectures, challenges, applications, future directions | 2021 | Referencia general si se escala a deep learning. |

## Limitaciones

- El corpus de noticias RSS es pequeño (SENT-001) y concentrado en fechas
  recientes; el gap de NVDA en MODEL-002 se explica por esto, no por el modelo.
- Los papers se obtuvieron de OpenAlex con búsqueda temática; no sustituyen
  una revisión bibliográfica exhaustiva.
