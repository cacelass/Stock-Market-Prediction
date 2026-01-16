# Stock Market Prediction

> "Existen numerosas formas de arruinarte. De entre ellas, crear un algoritmo de machine learning que trate de predecir el valor de las acciones en bolsa para luego invertir tu dinero, es una de las más rápidas y efectivas."

Este proyecto es una exploración técnica (y una lección de humildad) sobre la aplicación de Machine Learning en series temporales financieras. El objetivo principal no fue buscar rentabilidad inmediata, sino aprender enfrentándose a un problema real con "casi infinitos grados de libertad".

## El Problema: ¿Por qué es tan difícil?

Durante el desarrollo, se identificaron dos barreras principales que hacen que "rendimientos pasados no garanticen rendimientos futuros":

### 1. La Trampa del Overfitting (Desafíos Técnicos)
El primer obstáculo fue detectar que el modelo "hacía trampa". Resultados casi perfectos en entrenamiento suelen indicar que el modelo está memorizando ruido o mirando al futuro (data leakage). Si el error en train es drásticamente menor que la varianza del target, es señal inequívoca de overfitting.

Cualquier sistema de optimización tenderá a hacer esta "trampa" salvo que encontremos patrones robustos o forcemos restricciones.

### 2. El Caos del Mercado (Desafíos de Contexto)
Los mercados no son estáticos; responden a ciclos económicos y eventos impredecibles ("Cisnes Negros") que rompen cualquier patrón histórico:
* **Macroeconomía:** Crisis inmobiliaria (2008).
* **Eventos Globales:** Pandemia COVID-19 (2020).
* **Comportamiento Irracional:** Fenómenos como WallStreetBets y GameStop.

Cuesta imaginar cómo estructurar una recolección de datos que hubiera permitido anticipar la caída de marzo de 2020.

## Estrategias Implementadas

Para mitigar el ruido y tratar de construir un sistema que, al menos, minimice pérdidas, se exploraron las siguientes soluciones:

* **Simplificación del Target:** Transformación del problema a variable binaria (0: Baja, 1: Sube).
* **Ingeniería de Features:** Uso de ventanas temporales más amplias (Lags de semana, mes, trimestre) para capturar tendencias y no solo ruido diario.
* **Selección de Modelo:** Uso de Random Forest por su capacidad para manejar la varianza y reducir el sobreajuste frente a otros algoritmos.

---

## Organización del Proyecto

El proyecto sigue una estructura estandarizada de Data Science para garantizar reproducibilidad:


    ├── LICENSE
    ├── tasks.py           <- Tareas de automatización (ej. `notebook`).
    ├── README.md          <- Documentación principal.
    ├── install.md         <- Instrucciones de instalación.
    ├── pyproject.toml     <- Dependencias del proyecto.
    │
    ├── data
    │   ├── external       <- Datos de fuentes externas.
    │   ├── interim        <- Datos transformados intermedios.
    │   ├── processed      <- Datos finales para modelado.
    │   └── raw            <- Datos originales inmutables.
    │
    ├── models             <- Modelos serializados (.pkl) y reportes.
    ├── notebooks          <- Jupyter Notebooks (Naming convention: `1.0-jqp-descripcion`).
    ├── references         <- Diccionarios de datos y manuales.
    ├── reports            <- Análisis generados (HTML/PDF) y figuras.
    │
    └── inversion          <- Código fuente (Python Package).
        ├── data           <- Scripts de generación de dataset (`make_dataset.py`).
        ├── features       <- Ingeniería de variables (`build_features.py`).
        ├── models         <- Entrenamiento e inferencia (`train_model.py`, `predict_model.py`).
        ├── utils          <- Utilidades y rutas (`paths.py`).
        └── visualization  <- Gráficos (`visualize.py`).