from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from inversion.utils import paths


def train_rf_model(X_train, y_train, filename=None, random_state=42):
    """
    Entrena un modelo Random Forest y lo guarda.
    
    Args:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        filename: Nombre del archivo para guardar el modelo (opcional).
        random_state: Semilla para reproducibilidad (por defecto 42).
    """
    print(f"      ...Configurando RandomForest (semilla={random_state})")
    
    # Usamos el parámetro random_state en lugar de config.Random_State
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    
    model.fit(X_train, y_train)
    
    # Guardar el modelo
    # Si no nos pasan un nombre específico, usamos la ruta por defecto de paths
    save_path = paths.MODEL_FILE if filename is None else paths.MODELS_DIR / filename
    
    joblib.dump(model, save_path)
    return model

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo y devuelve MSE y R2.
    args:
        model: Modelo entrenado.
        X_test: Features de test.
        y_test: Target de test.
    returns:
        mse: Mean Squared Error
        r2: R² Score
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2