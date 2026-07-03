# ==========================================
# 5) MODELAGEM PREDITIVA (MACHINE LEARNING)
# ==========================================
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error, mean_squared_error, f1_score
from xgboost import XGBRegressor, XGBClassifier

FEATURE_COLUMNS: List[str] = ['prop_positivo', 'prop_negativo', 'prop_neutro', 'qtd_comentarios_coletados']


def prepare_features_and_targets(
    df_final: pd.DataFrame, feature_cols: List[str] = FEATURE_COLUMNS
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Extrai a matriz de features (X) e os alvos de regressão/classificação (y_reg, y_clf) de df_final."""
    X = df_final[feature_cols]
    y_reg = df_final['engagement_rate']
    y_clf = df_final['engagement_class']
    return X, y_reg, y_clf


def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula MAE, RMSE e R² entre valores reais e previstos."""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
    }


def compute_classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula Acurácia e F1 entre valores reais e previstos."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def train_and_evaluate(
    df_final: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLUMNS,
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Treina RandomForest de regressão/classificação com holdout e validação cruzada K-Fold."""
    # Variáveis de entrada (Features de Sentimento) e Saída (Alvo)
    X, y_reg, y_clf = prepare_features_and_targets(df_final, feature_cols)

    print("\n==========================================")
    print("             RESULTADOS DE ML             ")
    print("==========================================")

    results = {}

    # Se houver dados suficientes, treina os modelos
    if len(df_final) >= 2:
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
            X, y_reg, y_clf, test_size=test_size, random_state=random_state
        )

        # REGRESSÃO (Prever o número exato da taxa)
        reg_model = RandomForestRegressor(random_state=random_state)
        reg_model.fit(X_train, y_reg_train)
        y_pred_reg = reg_model.predict(X_test)
        reg_metrics = compute_regression_metrics(y_reg_test, y_pred_reg)
        mae, rmse, r2 = reg_metrics['mae'], reg_metrics['rmse'], reg_metrics['r2']
        print(f"Regressão (Random Forest) -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R² Score: {r2:.4f}")

        # CLASSIFICAÇÃO (Prever se vai ser Alto [1] ou Baixo [0] engajamento)
        clf_model = RandomForestClassifier(random_state=random_state)
        clf_model.fit(X_train, y_clf_train)
        y_pred_clf = clf_model.predict(X_test)
        clf_metrics = compute_classification_metrics(y_clf_test, y_pred_clf)
        acc, f1 = clf_metrics['accuracy'], clf_metrics['f1']
        print(f"Classificação (Random Forest) -> Acurácia: {acc:.4f} | F1: {f1:.4f}")

        # VALIDAÇÃO CRUZADA (K-Fold) sobre todo o dataset
        cv_splits = min(n_splits, len(df_final))
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        cv_r2_scores = cross_val_score(
            RandomForestRegressor(random_state=random_state), X, y_reg, cv=kf, scoring='r2'
        )
        cv_acc_scores = cross_val_score(
            RandomForestClassifier(random_state=random_state), X, y_clf, cv=kf, scoring='accuracy'
        )
        print(f"Validação Cruzada ({cv_splits}-Fold) -> R² médio: {cv_r2_scores.mean():.4f}")
        print(f"Validação Cruzada ({cv_splits}-Fold) -> Acurácia média: {cv_acc_scores.mean():.4f}")

        results = {
            'reg_model': reg_model,
            'clf_model': clf_model,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'accuracy': acc,
            'f1_score': f1,
            'cv_r2_mean': cv_r2_scores.mean(),
            'cv_accuracy_mean': cv_acc_scores.mean(),
        }
    else:
        print("Aviso: Adicione mais de 2 IDs de vídeos para calcular as métricas de Machine Learning.")

    return results


def compare_models(
    df_final: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLUMNS,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Treina e compara múltiplos modelos de regressão/classificação no mesmo split, retornando uma tabela."""
    columns = ['tipo', 'modelo', 'mae', 'rmse', 'r2', 'accuracy', 'f1']

    if len(df_final) < 2:
        return pd.DataFrame(columns=columns)

    X, y_reg, y_clf = prepare_features_and_targets(df_final, feature_cols)

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=test_size, random_state=random_state
    )

    regression_models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=random_state),
        'XGBoost': XGBRegressor(random_state=random_state),
    }
    classification_models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=random_state),
        'XGBoost': XGBClassifier(random_state=random_state, eval_metric='logloss'),
    }

    rows = []
    for name, model in regression_models.items():
        model.fit(X_train, y_reg_train)
        y_pred = model.predict(X_test)
        rows.append({
            'tipo': 'regressao',
            'modelo': name,
            **compute_regression_metrics(y_reg_test, y_pred),
            'accuracy': np.nan,
            'f1': np.nan,
        })

    for name, model in classification_models.items():
        model.fit(X_train, y_clf_train)
        y_pred = model.predict(X_test)
        rows.append({
            'tipo': 'classificacao',
            'modelo': name,
            'mae': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            **compute_classification_metrics(y_clf_test, y_pred),
        })

    return pd.DataFrame(rows, columns=columns)
