import os
import tempfile
import shutil
import torch


def save_checkpoint(model, optimizer, scheduler,early_stopper, epoch, current_batch_index, loss, path='checkpoint.pth'):
    """Guarda el estado completo del entrenamiento de forma atómica"""
    print(f"   💾 Guardando checkpoint para época {epoch}...")

    try:
        # 1. Crear el diccionario de checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'batch_idx': current_batch_index,
            'epochs_without_val_loss_improvement' : early_stopper.counter
        }

        # 2. Guardar en un archivo temporal primero
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"checkpoint_temp_{os.getpid()}.pth")

        # Guardar con formato que permita verificación
        torch.save(checkpoint, temp_path, _use_new_zipfile_serialization=True)

        # 3. Verificar que el archivo temporal se guardó correctamente
        try:
            # Intentar cargar el archivo temporal para verificar integridad
            test_checkpoint = torch.load(temp_path, map_location='cpu', weights_only=False)

            # Verificar que todas las claves necesarias están presentes
            required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict',
                           'scheduler_state_dict', 'loss', 'batch_idx', 'epochs_without_val_loss_improvement']
            for key in required_keys:
                if key not in test_checkpoint:
                    raise ValueError(f"Clave faltante en checkpoint: {key}")

            print(f"   ✅ Checkpoint verificado correctamente")

        except Exception as e:
            print(f"   ❌ Error al verificar checkpoint: {e}")
            # Limpiar archivo temporal corrupto
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        # 4. Si existe checkpoint anterior, crear backup
        backup_path = path + '.bak'
        if os.path.exists(path):
            try:
                shutil.copy2(path, backup_path)
                print(f"   📁 Backup creado: {backup_path}")
            except Exception as e:
                print(f"   ⚠️  No se pudo crear backup: {e}")

        # 5. Mover archivo temporal a ubicación final (operación atómica)
        shutil.move(temp_path, path)

        # 6. Verificar que el archivo final existe y tiene tamaño
        if os.path.exists(path) and os.path.getsize(path) > 100:  # Mínimo 100 bytes
            print(f"   ✅ Checkpoint guardado exitosamente en época {epoch}")
            
        else:
            raise IOError("El archivo final no se creó correctamente")

    except Exception as e:
        print(f"   ❌ Error crítico al guardar checkpoint: {e}")
        print(f"   ⚠️  El entrenamiento continuará sin guardar este checkpoint")

        # Intentar recuperar el backup si existe
        if os.path.exists(backup_path) and not os.path.exists(path):
            try:
                shutil.copy2(backup_path, path)
                print(f"   🔄 Restaurado checkpoint desde backup")
            except:
                pass


def load_checkpoint(model, optimizer, scheduler,early_stopper, path):
    """Carga el estado completo del entrenamiento con recuperación ante fallos"""
    print(f"   📥 Intentando cargar checkpoint desde {path}...")

    backup_path = path + '.bak'
    temp_path = os.path.join(tempfile.gettempdir(), f"checkpoint_temp_{os.getpid()}.pth")

    # Lista de posibles ubicaciones en orden de prioridad
    possible_paths = [
        path,              # Ubicación principal
        backup_path,       # Backup
        temp_path          # Archivo temporal (si existe)
    ]

    for checkpoint_path in possible_paths:
        if os.path.exists(checkpoint_path):
            try:
                print(f"   🔍 Probando {checkpoint_path}...")

                # Intentar cargar con diferentes métodos
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                except:
                    # Fallback si el método anterior falla
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')

                # Verificar estructura básica
                if not isinstance(checkpoint, dict):
                    print(f"   ❌ Formato inválido en {checkpoint_path}")
                    continue

                required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict',
                               'scheduler_state_dict', 'loss', 'batch_idx', 'epochs_without_val_loss_improvement']
                missing_keys = [k for k in required_keys if k not in checkpoint]

                if missing_keys:
                    print(f"   ❌ Claves faltantes en {checkpoint_path}: {missing_keys}")
                    continue

                # Cargar estados
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                early_stopper.counter = checkpoint['epochs_without_val_loss_improvement']
                start_epoch = checkpoint['epoch']
                current_batch_index = checkpoint['batch_idx']
                loss = checkpoint['loss']

                print(f"   ✅ Checkpoint cargado desde {checkpoint_path}")
                print(f"   📊 Continuando desde época {start_epoch}, batch: {current_batch_index}")
                print(f"   ⚠️Sin mejora en val_loss: {early_stopper.counter}/{early_stopper.patience} ")
                print(f"   📉 Pérdida anterior: {loss:.6f}")

                # Si cargamos desde backup, restaurar como checkpoint principal
                if checkpoint_path == backup_path and os.path.exists(backup_path):
                    try:
                        shutil.copy2(backup_path, path)
                        print(f"   🔄 Backup restaurado como checkpoint principal")
                    except:
                        pass

                return start_epoch, current_batch_index, loss

            except Exception as e:
                print(f"   ❌ Error al cargar {checkpoint_path}: {e}")
                continue

    # Si llegamos aquí, ningún checkpoint fue válido
    print("   ⚠️  No se encontró checkpoint válido. Comenzando desde cero.")

    # Limpiar archivos corruptos si existen
    for p in [path, backup_path, temp_path]:
        if os.path.exists(p):
            try:
                file_size = os.path.getsize(p)
                if file_size < 100:  # Archivo demasiado pequeño, probablemente corrupto
                    os.remove(p)
                    print(f"   🗑️  Eliminado archivo corrupto: {p} ({file_size} bytes)")
            except:
                pass

    return 0, 0, float('inf')
