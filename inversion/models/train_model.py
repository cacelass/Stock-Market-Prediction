from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
from inversion.utils import paths


def train_rf_model(X_train, y_train, filename=None, random_state=42) -> RandomForestClassifier:
    """
    Entrena un RandomForestClassifier y lo guarda en disco.

    Args:
        X_train      : Features de entrenamiento (array o DataFrame).
        y_train      : Target binario (0 = baja, 1 = sube).
        filename     : Nombre de archivo para el modelo (opcional).
        random_state : Semilla para reproducibilidad.

    Returns:
        Modelo entrenado.
    """
    print(f"      ...Configurando RandomForestClassifier (semilla={random_state})")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,           # limita profundidad para reducir overfitting
        min_samples_leaf=20,   # no divide con pocos ejemplos
        class_weight="balanced",  # compensa desbalance de clases
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    save_path = paths.MODEL_FILE if filename is None else paths.MODELS_DIR / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)

    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evalúa el modelo con métricas de clasificación.

    Returns:
        dict con accuracy, auc y el classification_report completo.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["Baja (0)", "Sube (1)"])

    return {"accuracy": acc, "auc": auc, "report": report}