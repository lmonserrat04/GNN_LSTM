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
    "lr":                  5e-4,
    "weight_decay":        0.05,
    "scheduler_step_size": 10,
    "scheduler_gamma":     0.4,
    "batch_size":          32,
    "n_epochs":            150,
    "max_grad_norm":       1.0,
    "patience":            100,
    "min_delta":           0.001,
}

# ──────────────────────────────────────────────────────────────────────────────
# Define qué parámetros explorar y en qué rango
# ──────────────────────────────────────────────────────────────────────────────
def objective(trial):
    cfg = {
        **FIXED,
        "pool_ratio":      trial.suggest_categorical("pool_ratio",      [0.5]),
        "hidden_channels": trial.suggest_categorical("hidden_channels",  [128]),
    }
    run_name = f"pool{cfg['pool_ratio']}_hid{cfg['hidden_channels']}"
    return run_training(cfg, run_name)

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        study_name="gnn_lstm_search",
        storage="sqlite:///search.db",
        load_if_exists=True,
    )

    import os, sqlite3
    from datetime import datetime

    def _run_name(params):
        return f"pool{params['pool_ratio']}_hid{params['hidden_channels']}"

    def _has_best_model(params):
        return os.path.exists(f"best_model_{_run_name(params)}.pth")

    def _has_checkpoint(params):
        return os.path.exists(f"checkpoint_{_run_name(params)}.pth")

    def _has_done(params):
        return os.path.exists(f"{_run_name(params)}.done")

    def _read_done(params):
        with open(f"{_run_name(params)}.done") as f:
            return float(f.read().strip())

    def _write_done(params, value):
        with open(f"{_run_name(params)}.done", 'w') as f:
            f.write(str(value))

    def _mark_complete_db(trial, value):
        conn = sqlite3.connect('search.db')
        cur = conn.cursor()
        cur.execute("SELECT trial_id FROM trials WHERE number=?", (trial.number,))
        trial_id = cur.fetchone()[0]
        cur.execute("UPDATE trials SET state='COMPLETE', datetime_complete=? WHERE trial_id=?",
                    (datetime.now().isoformat(), trial_id))
        cur.execute("SELECT COUNT(*) FROM trial_values WHERE trial_id=?", (trial_id,))
        if cur.fetchone()[0] == 0:
            cur.execute("INSERT INTO trial_values (trial_id, objective, value, value_type) VALUES (?,0,?,'FINITE')",
                        (trial_id, value))
        conn.commit()
        conn.close()

    def _mark_fail_db(trial):
        conn = sqlite3.connect('search.db')
        cur = conn.cursor()
        cur.execute("SELECT trial_id FROM trials WHERE number=?", (trial.number,))
        trial_id = cur.fetchone()[0]
        cur.execute("UPDATE trials SET state='FAIL', datetime_complete=? WHERE trial_id=?",
                    (datetime.now().isoformat(), trial_id))
        conn.commit()
        conn.close()

    # Limpiar trials en estado no terminal
    estados_sucios = [
        t for t in study.trials
        if t.state in (optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.FAIL)
    ]
    trials_interrumpidos = []
    for t in estados_sucios:
        rn = _run_name(t.params)
        if _has_done(t.params):
            valor = _read_done(t.params)
            _mark_complete_db(t, valor)
            print(f"🔧 Trial #{t.number} ({rn}) tiene .done → COMPLETE ({valor:.4f})")
        elif _has_best_model(t.params):
            valor = next(
                (x.value for x in study.trials
                 if x.state == optuna.trial.TrialState.COMPLETE
                 and x.params == t.params
                 and x.value is not None),
                None
            )
            if valor is not None:
                _mark_complete_db(t, valor)
                _write_done(t.params, valor)
                print(f"🔧 Trial #{t.number} ({rn}) tiene best_model → COMPLETE ({valor:.4f})")
            else:
                trials_interrumpidos.append(t)
        elif _has_checkpoint(t.params):
            trials_interrumpidos.append(t)
        else:
            _mark_fail_db(t)
            print(f"🔧 Trial #{t.number} ({rn}) sin archivos → FAIL")

    study = optuna.load_study(study_name="gnn_lstm_search", storage="sqlite:///search.db")

    if trials_interrumpidos:
        trial_interrumpido = max(trials_interrumpidos, key=lambda t: t.number)
        print(f"🔄 Reanudando trial #{trial_interrumpido.number} ({_run_name(trial_interrumpido.params)})...")
        best_loss = run_training({**FIXED,
                                   "pool_ratio": trial_interrumpido.params["pool_ratio"],
                                   "hidden_channels": trial_interrumpido.params["hidden_channels"]},
                                  _run_name(trial_interrumpido.params))
        print(f"✅ Trial #{trial_interrumpido.number} completado — loss={best_loss:.4f}")

    # Continuar con los trials restantes
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print("\n" + "="*50)
    print("  MEJOR CONFIGURACIÓN ENCONTRADA")
    print("="*50)
    for k, v in study.best_params.items():
        print(f"  {k} = {v}")
    print(f"\n  Mejor loss: {study.best_value:.4f}")
    print(f"  Trial #:    {study.best_trial.number}")
