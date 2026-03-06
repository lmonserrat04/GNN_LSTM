import os
import tempfile
import shutil
import torch


def save_checkpoint(model, optimizer, scheduler, early_stopper, epoch, current_batch_index, loss, path='checkpoint.pth'):
    print(f"   💾 Guardando checkpoint para época {epoch}...")
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'batch_idx': current_batch_index,
            'early_stopper_counter': early_stopper.counter,
            'early_stopper_best_loss': early_stopper.best_loss,
        }

        # Guardar en temporal primero
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"checkpoint_temp_{os.getpid()}.pth")
        torch.save(checkpoint, temp_path, _use_new_zipfile_serialization=True)

        # Verificar integridad del temporal
        try:
            test = torch.load(temp_path, map_location='cpu', weights_only=False)
            required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict',
                             'scheduler_state_dict', 'loss', 'batch_idx',
                             'early_stopper_counter', 'early_stopper_best_loss']
            for key in required_keys:
                if key not in test:
                    raise ValueError(f"Clave faltante en checkpoint: {key}")
            print(f"   ✅ Checkpoint verificado correctamente")
        except Exception as e:
            print(f"   ❌ Error al verificar checkpoint: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        # Backup del checkpoint anterior
        backup_path = path + '.bak'
        if os.path.exists(path):
            try:
                shutil.copy2(path, backup_path)
                print(f"   📁 Backup creado: {backup_path}")
            except Exception as e:
                print(f"   ⚠️ No se pudo crear backup: {e}")

        # Mover temporal a ubicación final (operación atómica)
        shutil.move(temp_path, path)

        if os.path.exists(path) and os.path.getsize(path) > 100:
            print(f"   ✅ Checkpoint guardado exitosamente en época {epoch}")
        else:
            raise IOError("El archivo final no se creó correctamente")

    except Exception as e:
        print(f"   ❌ Error crítico al guardar checkpoint: {e}")
        print(f"   ⚠️ El entrenamiento continuará sin guardar este checkpoint")
        backup_path = path + '.bak'
        if os.path.exists(backup_path) and not os.path.exists(path):
            try:
                shutil.copy2(backup_path, path)
                print(f"   🔄 Restaurado checkpoint desde backup")
            except:
                pass


def load_checkpoint(model, optimizer, scheduler, early_stopper, path):
    print(f"   📥 Intentando cargar checkpoint desde {path}...")
    backup_path = path + '.bak'
    temp_path = os.path.join(tempfile.gettempdir(), f"checkpoint_temp_{os.getpid()}.pth")

    possible_paths = [path, backup_path, temp_path]

    for checkpoint_path in possible_paths:
        if not os.path.exists(checkpoint_path):
            continue
        try:
            print(f"   🔍 Probando {checkpoint_path}...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            except:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if not isinstance(checkpoint, dict):
                print(f"   ❌ Formato inválido en {checkpoint_path}")
                continue

            required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict',
                             'scheduler_state_dict', 'loss', 'batch_idx',
                             'early_stopper_counter', 'early_stopper_best_loss']
            missing_keys = [k for k in required_keys if k not in checkpoint]
            if missing_keys:
                print(f"   ❌ Claves faltantes en {checkpoint_path}: {missing_keys}")
                continue

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            early_stopper.counter   = checkpoint['early_stopper_counter']
            early_stopper.best_loss = checkpoint['early_stopper_best_loss']
            start_epoch  = checkpoint['epoch']
            batch_idx    = checkpoint['batch_idx']
            loss         = checkpoint['loss']

            print(f"   ✅ Checkpoint cargado desde {checkpoint_path}")
            print(f"   📊 Época {start_epoch} | Batch {batch_idx}")
            print(f"   ⚠️ Sin mejora: {early_stopper.counter}/{early_stopper.patience} | Mejor val_loss: {early_stopper.best_loss:.4f}")
            return start_epoch, batch_idx, loss

        except Exception as e:
            print(f"   ❌ Error cargando {checkpoint_path}: {e}")
            continue

    # Limpiar archivos corruptos pequeños
    for p in [path, backup_path, temp_path]:
        if os.path.exists(p):
            try:
                if os.path.getsize(p) < 100:
                    os.remove(p)
                    print(f"   🗑️ Eliminado archivo corrupto: {p}")
            except:
                pass

    print("   ⚠️ No se encontró checkpoint válido. Comenzando desde cero.")
    return 0, 0, float('inf')
