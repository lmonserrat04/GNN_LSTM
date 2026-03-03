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
    "scheduler_step_size": 10,
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

    # ── Limpiar trials huérfanos y reanudar interrumpidos ────────────────────
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

    # Limpiar todos los trials en estado no terminal
    estados_sucios = [
        t for t in study.trials
        if t.state in (optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.FAIL)
    ]

    trials_interrumpidos = []

    for t in estados_sucios:
        rn = _run_name(t.params)
        if _has_done(t.params):
            # Completó correctamente — marcar COMPLETE con valor del .done
            valor = _read_done(t.params)
            _mark_complete_db(t, valor)
            print(f"🔧 Trial #{t.number} ({rn}) tiene .done → COMPLETE ({valor:.4f})")
        elif _has_best_model(t.params):
            # Tiene best_model pero no .done — buscar valor en trials COMPLETE anteriores
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
                # No hay referencia — reentrenar
                trials_interrumpidos.append(t)
        elif _has_checkpoint(t.params):
            # Interrumpido a mitad — reanudar desde checkpoint
            trials_interrumpidos.append(t)
        else:
            # Sin nada — marcar FAIL
            _mark_fail_db(t)
            print(f"🔧 Trial #{t.number} ({rn}) sin archivos → FAIL")

    # Recargar study para reflejar cambios en DB
    study = optuna.load_study(study_name="gnn_lstm_search", storage="sqlite:///search.db")

    if trials_interrumpidos:
        trial_interrumpido = max(trials_interrumpidos, key=lambda t: t.number)
        params = trial_interrumpido.params
        run_name = _run_name(params)

        print(f"\n⚡ Trial #{trial_interrumpido.number} interrumpido detectado")
        print(f"   pool_ratio={params['pool_ratio']}  hidden_channels={params['hidden_channels']}")
        print(f"   Reanudando desde checkpoint...\n")

        cfg = {**FIXED, **params}
        best_loss = run_training(cfg, run_name=run_name)

        # Escribir .done como señal de completado
        _write_done(params, best_loss)

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
