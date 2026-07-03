# ==========================================
# 6) EXPORTAR RESULTADOS PARA O TCC
# ==========================================
import os
from typing import Any, Dict

import pandas as pd


def save_outputs(df_comments: pd.DataFrame, df_final: pd.DataFrame, output_dir: str = "outputs") -> None:
    """Salva os comentários processados e o dataset final como CSV em output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    df_comments.to_csv(f"{output_dir}/comentarios_processados.csv", index=False, encoding="utf-8-sig")
    df_final.to_csv(f"{output_dir}/dataset_final_videos.csv", index=False, encoding="utf-8-sig")
    print("\n[SUCESSO] Planilhas salvas com sucesso na pasta 'outputs'!")


def save_metrics(metrics: Dict[str, Any], output_dir: str = "outputs/metrics") -> None:
    """Salva um dicionário de métricas como uma linha de CSV em output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(f"{output_dir}/metrics.csv", index=False, encoding="utf-8-sig")
    print(f"[SUCESSO] Métricas salvas em '{output_dir}/metrics.csv'!")


def save_model_comparison(df_comparison: pd.DataFrame, output_dir: str = "outputs/metrics") -> None:
    """Salva a tabela de comparação de modelos como CSV em output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    df_comparison.to_csv(f"{output_dir}/comparacao_modelos.csv", index=False, encoding="utf-8-sig")
    print(f"[SUCESSO] Comparação de modelos salva em '{output_dir}/comparacao_modelos.csv'!")
