"""
search.py
=========
Búsqueda de hiperparámetros con Optuna.

Uso:
    python search.py
"""

import optuna
from train import run_training


# ──────────────────────────────────────────────────────────────────────────────
# Parámetros fijos (no se buscan)
# ──────────────────────────────────────────────────────────────────────────────
FIXED = {
    "lr":                  1e-3,
    "weight_decay":        1e-2,
    "scheduler_step_size": 20,
    "scheduler_gamma":     0.4,
    "batch_size":          32,
    "n_epochs":            150,
    "max_grad_norm":       1.0,
    "patience":            30,
    "min_delta":           0.001,
}


# ──────────────────────────────────────────────────────────────────────────────
# Define qué parámetros explorar y en qué rango
# ──────────────────────────────────────────────────────────────────────────────
def objective(trial):
    cfg = {
        **FIXED,
        "pool_ratio":      trial.suggest_categorical("pool_ratio",      [0.15, 0.30, 0.50]),
        "hidden_channels": trial.suggest_categorical("hidden_channels",  [32, 64, 128]),
    }
    # Nombre legible para checkpoint: pool0.15_hid64, pool0.30_hid128, etc.
    run_name = f"pool{cfg['pool_ratio']}_hid{cfg['hidden_channels']}"
    return run_training(cfg, run_name)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    study = optuna.create_study(
        direction="minimize",
        study_name="gnn_lstm_search",
        storage="sqlite:///search.db",  # guarda todo en disco
        load_if_exists=True,            # si ya existe, continúa desde donde estaba
    )

    # ── Reanudar trial interrumpido si existe ─────────────────────
    # Un trial queda como FAIL si el proceso se cortó a mitad
    trials_fallidos = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    if trials_fallidos:
        trial_interrumpido = trials_fallidos[-1]
        params = trial_interrumpido.params
        print(f"\n⚡ Trial #{trial_interrumpido.number} interrumpido detectado")
        print(f"   pool_ratio={params['pool_ratio']}  hidden_channels={params['hidden_channels']}")
        print(f"   Reanudando desde checkpoint...\n")

        cfg = {**FIXED, **params}
        run_name = f"pool{cfg['pool_ratio']}_hid{cfg['hidden_channels']}"

        # run_training detecta el checkpoint existente y retoma desde la época/batch exactos
        best_loss = run_training(cfg, run_name=run_name)

        # Registrar el resultado del trial interrumpido en Optuna
        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions=trial_interrumpido.distributions,
                value=best_loss,
            )
        )
        print(f"✅ Trial #{trial_interrumpido.number} completado — loss={best_loss:.4f}")

    # ── Continuar con los trials restantes ────────────────────────
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    # ── Resultados ────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  MEJOR CONFIGURACIÓN ENCONTRADA")
    print("="*50)
    for k, v in study.best_params.items():
        print(f"  {k} = {v}")
    print(f"\n  Mejor loss: {study.best_value:.4f}")
    print(f"  Trial #:    {study.best_trial.number}")
