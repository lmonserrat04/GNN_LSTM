import torch
import gc
from collections import Counter


# ============================================================================
# FUNCIÓN PRINCIPAL: Limpieza completa DESPUÉS de backward
# ============================================================================

# VERSIÓN SIMPLIFICADA para tu caso:
def cleanup_batch_simple(time_series_batch, lw_matrixes_sequence_batch, 
                         preds_batch, pool_losses_batch, labels_batch,
                         model, optimizer, extra_vars=None):
    """
    Limpieza simple para listas de tensores (sin objetos Data)
    """
    # Limpiar variables extra
    if extra_vars:
        for var in extra_vars.values():
            del var
    
    # Limpiar listas - Python las libera automáticamente
    _deep_clean_list(time_series_batch)
    _deep_clean_lw_matrixes_sequence_batch(lw_matrixes_sequence_batch)
    _deep_clean_list(preds_batch)
    _deep_clean_list(pool_losses_batch)
    del labels_batch
    
    # Limpiar gradientes
    optimizer.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.grad = None
    
    # GC agresivo
    import gc
    gc.collect()
    
# ============================================================================
# FUNCIÓN: Limpieza de gradientes del modelo
# ============================================================================

def cleanup_model_gradients(model, optimizer):
    """
    Limpia todos los gradientes del modelo de forma agresiva.

    Args:
        model: Modelo de PyTorch
        optimizer: Optimizador
    """
    optimizer.zero_grad(set_to_none=True)

    for param in model.parameters():
        param.grad = None

    # Limpiar hooks si existen
    for module in model.modules():
        if hasattr(module, '_backward_hooks'):
            module._backward_hooks.clear()
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
        if hasattr(module, '_forward_pre_hooks'):
            module._forward_pre_hooks.clear()


# ============================================================================
# FUNCIONES INTERNAS
# ============================================================================

def _deep_clean_lw_matrixes_sequence_batch(lw_matrixes_sequence_batch):
    """
    Limpia profundamente una lista de secuencias de matrices lw.
    
    """

    while lw_matrixes_sequence_batch:
        lw_matrix_list = lw_matrixes_sequence_batch.pop()

        while lw_matrix_list:
            lw = lw_matrix_list.pop()

            del lw
        
        del lw_matrix_list
    
    del lw_matrixes_sequence_batch

            


    


def _deep_clean_list(lst):
    """
    Limpia profundamente una lista y todos sus elementos.
    """
    if lst is None:
        return

    while lst:
        item = lst.pop()
        del item

    del lst


def _aggressive_gc():
    """
    Garbage collection agresivo con múltiples pasadas.
    """
    # Primera pasada
    gc.collect()
    #torch.cuda.empty_cache()

    # Sincronizar GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Segunda pasada (crítica)
    gc.collect()

    # Tercera pasada de cache
    #torch.cuda.empty_cache()


# ============================================================================
# FUNCIÓN: Preparar grafos con .detach()
# ============================================================================

# def prepare_graph_batch(X_lw_matrixes, idxs_for_batch, device):
#     """
#     Prepara un batch de grafos con .detach() para evitar memory leaks.

#     Args:
#         X_lw_matrixes: Lista completa de grafos del dataset
#         idxs_for_batch: Índices del batch actual
#         device: Device de PyTorch (cuda o cpu)

#     Returns:
#         Lista de listas de Data objects preparados
#     """
#     from torch_geometric.data import Data

#     lw_matrixes_sequence_batch = []

#     for idx in idxs_for_batch:
#         subject_graphs = []

#         for g in X_lw_matrixes[idx]:
#             # CRÍTICO: .detach() para romper referencias
#             new_graph = Data(
#                 x=g.x.detach().clone().to(device),
#                 edge_index=g.edge_index.detach().clone().to(device)
#             )
#             subject_graphs.append(new_graph)

#         lw_matrixes_sequence_batch.append(subject_graphs)

#     return lw_matrixes_sequence_batch


# ============================================================================
# FUNCIÓN DE DIAGNÓSTICO
# ============================================================================

def diagnose_memory(verbose=False):
    """
    Diagnóstico rápido de memoria GPU.

    Args:
        verbose: Si True, muestra información detallada

    Returns:
        Dict con estadísticas de memoria
    """
    stats = {
        'allocated_mb': torch.cuda.memory_allocated() / (1024**2),
        'reserved_mb': torch.cuda.memory_reserved() / (1024**2),
        'gpu_tensors': 0,
        'data_objects': 0,
        'large_lists': 0
    }

    # Contar objetos
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                stats['gpu_tensors'] += 1

            if hasattr(obj, 'x') and hasattr(obj, 'edge_index'):
                stats['data_objects'] += 1

            if isinstance(obj, list) and len(obj) > 50:
                stats['large_lists'] += 1
        except:
            pass

    if verbose:
        print(f"\n🔍 MEMORIA GPU:")
        print(f"  Allocated: {stats['allocated_mb']:.2f} MB")
        print(f"  Reserved:  {stats['reserved_mb']:.2f} MB")
        print(f"  Tensores:  {stats['gpu_tensors']}")
        print(f"  Data objs: {stats['data_objects']}")
        print(f"  Listas:    {stats['large_lists']}")

        if stats['data_objects'] > 10:
            print(f"  ⚠️ Muchos Data objects sin liberar")
        if stats['large_lists'] > 5:
            print(f"  ⚠️ Listas grandes sin limpiar")

    return stats


# ============================================================================
# FUNCIÓN AVANZADA: Encontrar referencias (solo para debug)
# ============================================================================

def find_memory_leaks(target_shape=None, top_n=5):
    """
    Encuentra qué objetos están causando memory leaks.
    Solo usar para debugging, es lento.

    Args:
        target_shape: Shape específico a buscar, ej: [2, 39800]
        top_n: Cuántos resultados mostrar

    Returns:
        Dict con información de leaks
    """
    print(f"\n🔬 Buscando memory leaks...")

    tensor_refs = {}

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                shape_str = str(list(obj.shape))

                if target_shape and shape_str != str(target_shape):
                    continue

                if shape_str not in tensor_refs:
                    tensor_refs[shape_str] = {'count': 0, 'refs': []}

                tensor_refs[shape_str]['count'] += 1

                # Analizar referencias
                referrers = gc.get_referrers(obj)
                for ref in referrers[:3]:
                    ref_type = type(ref).__name__
                    if isinstance(ref, list):
                        ref_info = f"list(len={len(ref)})"
                    elif isinstance(ref, dict):
                        ref_info = "dict"
                    else:
                        ref_info = ref_type
                    tensor_refs[shape_str]['refs'].append(ref_info)
        except:
            pass

    # Mostrar resultados
    print(f"\n📊 TENSORES CON MÁS REFERENCIAS:\n")
    sorted_shapes = sorted(tensor_refs.items(), key=lambda x: x[1]['count'], reverse=True)

    for shape, info in sorted_shapes[:top_n]:
        print(f"Shape {shape}: {info['count']} tensores")
        ref_counts = Counter(info['refs'])
        for ref_type, count in ref_counts.most_common(3):
            print(f"  • {ref_type}: {count}x")

    return tensor_refs


# ============================================================================
# FUNCIÓN: Verificación post-limpieza
# ============================================================================

def verify_cleanup(expected_tensors=200, warn=True):
    """
    Verifica que la limpieza fue efectiva.

    Args:
        expected_tensors: Número esperado de tensores después de limpieza
        warn: Si True, muestra advertencias

    Returns:
        True si la limpieza fue exitosa
    """
    stats = diagnose_memory(verbose=False)

    success = (
        stats['gpu_tensors'] < expected_tensors * 1.5 and
        stats['data_objects'] < 10 and
        stats['large_lists'] < 5
    )

    if warn and not success:
        print(f"\n⚠️ LIMPIEZA INCOMPLETA:")
        print(f"  Tensores: {stats['gpu_tensors']} (esperado: ~{expected_tensors})")
        print(f"  Data objects: {stats['data_objects']} (esperado: <10)")
        print(f"  Listas grandes: {stats['large_lists']} (esperado: <5)")

    return success


# ============================================================================
# INFORMACIÓN DEL MÓDULO
# ============================================================================

__version__ = "2.0.0"
__all__ = [
    'cleanup_batch_simple',
    'cleanup_model_gradients',
    'prepare_graph_batch',
    'diagnose_memory',
    'find_memory_leaks',
    'verify_cleanup'
]