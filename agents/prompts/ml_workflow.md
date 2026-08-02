# ML Workflow — Ciclo de modelo

## Pipeline
```
make train  →  make predict
data/interim/   models/ + reports/
```

## Según tipo de ML


### Supervisado — Clasificacion
| Paso | Comando | Agente |
|------|---------|--------|
| Entrenar | `make train` | `ml` (inspect_model tras entrenar) |
| Evaluar | `make predict` | `ml` (overfitting, comparación) + `graph` (figuras) |




  Métricas: accuracy, F1, precision, recall, ROC-AUC, matriz confusión
  
  



Modelos disponibles: RandomForest



## Agente `ml` — acciones clave
- `inspect_model` — tipo de estimador, parámetros, features
- `check_overfitting` — gap train/test, necesita threshold
- `feature_importance` — ranking de features (modelos árbol)
- `model_comparison` — ranking por tamaño, tipo, params
- `list_models` — descubre modelos en `models/`


## Agente `graph` — figuras
- `audit_figures` — detecta figuras vacías, aspect ratio incorrecto
- Se ejecuta automáticamente en pipelines develop/fix

## Problemas comunes
- Overfitting → gap train/test > threshold. Solución: regularización, más datos, early stopping
- Modelo no encontrado → `make train` no ejecutado o falló
- Métricas malas → revisar features, más ingeniería, probar otros modelos
- 
