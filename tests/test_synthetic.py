import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression

from tests.fixtures.synthetic import (
    make_leaky_dataset,
    make_synthetic_dataset,
    sample_size_status,
)


def test_recovers_effect():
    """Com efeito real (effect_size=0.4), a correlação entre prop_positivo
    e engagement_rate deve ser positiva e estatisticamente significativa."""
    _, df_videos = make_synthetic_dataset(n_videos=500, effect_size=0.4, seed=42)

    rho, p_value = stats.spearmanr(df_videos["prop_positivo"], df_videos["engagement_rate"])

    assert rho > 0
    assert p_value < 0.05


def test_no_false_positive():
    """Com effect_size=0.0, o sentimento NÃO tem efeito sobre o engajamento.
    Um pipeline honesto não deve rejeitar H0 (p > 0.05). Este é o teste
    mais importante do repositório: se ele falhar, o pipeline está
    inventando efeito onde não existe."""
    _, df_videos = make_synthetic_dataset(n_videos=500, effect_size=0.0, seed=42)

    _, p_value = stats.spearmanr(df_videos["prop_positivo"], df_videos["engagement_rate"])

    assert p_value > 0.05


def test_leakage_detected():
    """Com make_leaky_dataset, uma feature que é função direta do alvo
    permite prever o alvo com R² absurdamente alto -- o sintoma que o
    teste de vazamento da Tarefa 2 deve pegar em dados reais."""
    _, df_videos = make_leaky_dataset(n_videos=200, effect_size=0.4, seed=42)

    X = df_videos[["qtd_comentarios_coletados"]]
    y = df_videos["engagement_rate"]

    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)

    assert r2 > 0.95


def test_small_n_warns():
    """Com n=10 (entre 8 e 50), a amostra é pequena demais para ser
    reportada como confiável, mas grande o suficiente para treinar."""
    status = sample_size_status(10)

    assert status["confiavel"] is False

    with pytest.raises(RuntimeError):
        sample_size_status(5)
