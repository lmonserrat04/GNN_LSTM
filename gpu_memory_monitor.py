import torch
import gc
import tracemalloc
from collections import defaultdict
import pandas as pd

class GPUMemoryMonitor:
    """
    Monitor completo de memoria GPU para debugging de memory leaks
    """

    def __init__(self):
        self.snapshots = []
        self.tensor_history = []

    def get_tensor_info(self):
        """Obtiene información detallada de todos los tensores en GPU"""
        tensors_info = []

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if obj.is_cuda:
                        tensor_info = {
                            'type': type(obj).__name__,
                            'size': obj.size(),
                            'dtype': obj.dtype,
                            'device': obj.device,
                            'memory_mb': obj.element_size() * obj.nelement() / (1024**2),
                            'requires_grad': obj.requires_grad,
                            'grad_fn': str(obj.grad_fn) if obj.grad_fn else None,
                            'shape_str': str(list(obj.shape)),
                        }
                        tensors_info.append(tensor_info)
            except:
                pass

        return tensors_info

    def get_cuda_memory_summary(self):
        """Resumen completo de memoria CUDA"""
        if not torch.cuda.is_available():
            return "CUDA no disponible"

        summary = {
            'allocated_mb': torch.cuda.memory_allocated() / (1024**2),
            'reserved_mb': torch.cuda.memory_reserved() / (1024**2),
            'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024**2),
            'max_reserved_mb': torch.cuda.max_memory_reserved() / (1024**2),
            'cached_mb': torch.cuda.memory_reserved() / (1024**2) - torch.cuda.memory_allocated() / (1024**2),
        }
        return summary

    def snapshot(self, label=""):
        """Toma una instantánea del estado de memoria"""
        snapshot = {
            'label': label,
            'memory': self.get_cuda_memory_summary(),
            'tensor_count': len([obj for obj in gc.get_objects() if torch.is_tensor(obj) and obj.is_cuda]),
            'tensors': self.get_tensor_info()
        }
        self.snapshots.append(snapshot)
        return snapshot

    def compare_snapshots(self, idx1=-2, idx2=-1):
        """Compara dos snapshots y muestra las diferencias"""
        if len(self.snapshots) < 2:
            print("⚠️ Se necesitan al menos 2 snapshots para comparar")
            return

        snap1 = self.snapshots[idx1]
        snap2 = self.snapshots[idx2]

        print(f"\n{'='*80}")
        print(f"📊 COMPARACIÓN DE MEMORIA: {snap1['label']} vs {snap2['label']}")
        print(f"{'='*80}")

        # Diferencias en memoria
        mem1 = snap1['memory']
        mem2 = snap2['memory']

        print("\n🔍 CAMBIOS EN MEMORIA:")
        print(f"  Allocated:  {mem1['allocated_mb']:.2f} MB → {mem2['allocated_mb']:.2f} MB "
              f"(Δ {mem2['allocated_mb']-mem1['allocated_mb']:+.2f} MB)")
        print(f"  Reserved:   {mem1['reserved_mb']:.2f} MB → {mem2['reserved_mb']:.2f} MB "
              f"(Δ {mem2['reserved_mb']-mem1['reserved_mb']:+.2f} MB)")
        print(f"  Cached:     {mem1['cached_mb']:.2f} MB → {mem2['cached_mb']:.2f} MB "
              f"(Δ {mem2['cached_mb']-mem1['cached_mb']:+.2f} MB)")
        print(f"  Tensors:    {snap1['tensor_count']} → {snap2['tensor_count']} "
              f"(Δ {snap2['tensor_count']-snap1['tensor_count']:+d})")

        # Análisis de tensores nuevos
        tensors1_shapes = [t['shape_str'] for t in snap1['tensors']]
        tensors2_shapes = [t['shape_str'] for t in snap2['tensors']]

        # Contar tensores por forma
        from collections import Counter
        count1 = Counter(tensors1_shapes)
        count2 = Counter(tensors2_shapes)

        print("\n🆕 TENSORES NUEVOS (que no se liberaron):")
        new_shapes = set(count2.keys()) - set(count1.keys())
        increased_shapes = {k: count2[k] - count1.get(k, 0)
                           for k in count2.keys()
                           if count2[k] > count1.get(k, 0)}

        if increased_shapes:
            # Calcular memoria por tipo
            shape_memory = defaultdict(float)
            for tensor in snap2['tensors']:
                shape_str = tensor['shape_str']
                if shape_str in increased_shapes:
                    shape_memory[shape_str] += tensor['memory_mb']

            # Ordenar por memoria
            sorted_shapes = sorted(shape_memory.items(), key=lambda x: x[1], reverse=True)

            for shape, mem_mb in sorted_shapes[:10]:  # Top 10
                count_diff = increased_shapes[shape]
                total_mem = mem_mb
                print(f"  • Shape {shape}: +{count_diff} tensores, {total_mem:.2f} MB")
        else:
            print("  ✅ No hay tensores nuevos sin liberar")

        # Gradientes acumulados
        grads1 = [t for t in snap1['tensors'] if t.get('grad_fn')]
        grads2 = [t for t in snap2['tensors'] if t.get('grad_fn')]

        if len(grads2) > len(grads1):
            print(f"\n⚠️ GRADIENTES: {len(grads1)} → {len(grads2)} (+{len(grads2)-len(grads1)})")
            print("  Posible acumulación de grafo computacional")

        print(f"{'='*80}\n")

    def print_detailed_report(self):
        """Imprime un reporte detallado de la memoria actual"""
        tensors = self.get_tensor_info()
        memory = self.get_cuda_memory_summary()

        print(f"\n{'='*80}")
        print(f"📋 REPORTE DETALLADO DE MEMORIA GPU")
        print(f"{'='*80}")

        print(f"\n💾 MEMORIA TOTAL:")
        print(f"  Allocated: {memory['allocated_mb']:.2f} MB")
        print(f"  Reserved:  {memory['reserved_mb']:.2f} MB")
        print(f"  Cached:    {memory['cached_mb']:.2f} MB")
        print(f"  Max Alloc: {memory['max_allocated_mb']:.2f} MB")

        print(f"\n📦 TENSORES EN GPU: {len(tensors)}")

        # Agrupar por forma
        from collections import defaultdict
        shapes_summary = defaultdict(lambda: {'count': 0, 'memory': 0, 'with_grad': 0})

        for t in tensors:
            shape_str = t['shape_str']
            shapes_summary[shape_str]['count'] += 1
            shapes_summary[shape_str]['memory'] += t['memory_mb']
            if t['requires_grad'] or t['grad_fn']:
                shapes_summary[shape_str]['with_grad'] += 1

        # Ordenar por memoria
        sorted_shapes = sorted(shapes_summary.items(),
                              key=lambda x: x[1]['memory'],
                              reverse=True)

        print(f"\n🔝 TOP 15 FORMAS DE TENSORES (por memoria):")
        print(f"{'Shape':<30} {'Count':<8} {'Memory (MB)':<12} {'With Grad':<10}")
        print(f"{'-'*70}")

        for shape, info in sorted_shapes[:15]:
            print(f"{shape:<30} {info['count']:<8} {info['memory']:<12.2f} {info['with_grad']:<10}")

        # Buscar posibles leaks
        print(f"\n⚠️ POSIBLES MEMORY LEAKS:")

        # Tensores con gradientes
        grad_tensors = [t for t in tensors if t['grad_fn']]
        if grad_tensors:
            total_grad_mem = sum(t['memory_mb'] for t in grad_tensors)
            print(f"  • {len(grad_tensors)} tensores con grad_fn ({total_grad_mem:.2f} MB)")
            print(f"    Estos mantienen el grafo computacional vivo")

        # Tensores muy grandes
        large_tensors = [t for t in tensors if t['memory_mb'] > 10]
        if large_tensors:
            total_large_mem = sum(t['memory_mb'] for t in large_tensors)
            print(f"  • {len(large_tensors)} tensores >10MB ({total_large_mem:.2f} MB)")
            for t in sorted(large_tensors, key=lambda x: x['memory_mb'], reverse=True)[:5]:
                print(f"    - {t['shape_str']}: {t['memory_mb']:.2f} MB, grad_fn={t['grad_fn']}")

        print(f"{'='*80}\n")

    def export_to_csv(self, filename='gpu_memory_log.csv'):
        """Exporta el historial a CSV"""
        if not self.snapshots:
            print("⚠️ No hay snapshots para exportar")
            return

        data = []
        for snap in self.snapshots:
            row = {
                'label': snap['label'],
                'allocated_mb': snap['memory']['allocated_mb'],
                'reserved_mb': snap['memory']['reserved_mb'],
                'cached_mb': snap['memory']['cached_mb'],
                'tensor_count': snap['tensor_count'],
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"✅ Historial exportado a {filename}")

    def reset_peak_stats(self):
        """Resetea las estadísticas de pico de memoria"""
        torch.cuda.reset_peak_memory_stats()
        print("✅ Estadísticas de pico reseteadas")


def monitor_batch_memory(monitor, batch_idx, epoch, stage=""):
    """
    Función auxiliar para monitorear memoria en cada batch

    Uso:
        monitor = GPUMemoryMonitor()

        # Antes del batch
        monitor_batch_memory(monitor, batch_idx, epoch, "PRE")

        # ... procesar batch ...

        # Después del batch
        monitor_batch_memory(monitor, batch_idx, epoch, "POST")
        monitor.compare_snapshots()  # Comparar PRE vs POST
    """
    label = f"Epoch{epoch}_Batch{batch_idx}_{stage}"
    monitor.snapshot(label)

    mem = monitor.get_cuda_memory_summary()
    print(f"\n{'🔍' if stage=='PRE' else '✅'} {label}")
    print(f"  Allocated: {mem['allocated_mb']:.2f} MB | Reserved: {mem['reserved_mb']:.2f} MB")
    print(f"  Cached: {mem['cached_mb']:.2f} MB | Tensors en GPU: {monitor.snapshots[-1]['tensor_count']}")


def cleanup_gpu():
    """Limpieza agresiva de memoria GPU"""
    print("\n🧹 LIMPIANDO MEMORIA GPU...")

    # Contar tensores antes
    tensors_before = len([obj for obj in gc.get_objects()
                         if torch.is_tensor(obj) and obj.is_cuda])
    mem_before = torch.cuda.memory_allocated() / (1024**2)

    # Limpiar
    gc.collect()
    #torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Contar después
    tensors_after = len([obj for obj in gc.get_objects()
                        if torch.is_tensor(obj) and obj.is_cuda])
    mem_after = torch.cuda.memory_allocated() / (1024**2)

    print(f"  Tensores: {tensors_before} → {tensors_after} (liberados: {tensors_before-tensors_after})")
    print(f"  Memoria: {mem_before:.2f} MB → {mem_after:.2f} MB (liberados: {mem_before-mem_after:.2f} MB)")
