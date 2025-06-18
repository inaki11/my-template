# Ejecutar Sweeps con Weights & Biases

Sigue estos pasos para lanzar y ejecutar un sweep en wandb:

0. **Dependencias**

```python
pip install wandb scikit-learn tqdm tensorboard rtdl_num_embeddings omegaconf holidays
```

1. **Crear el sweep:**
activa el entorno de conda y situate en la raiz del repositorio

```bash
wandb sweep configs/sweep_periodical_mlp.yaml
```
Este comando devolverá un sweep_id

2. **Ejecutar el agente para el sweep:**

Te devuelve el comando para ejecutar sweeps, algo del estilo wandb agent inaki/my-template/874plwyb

```bash
wandb agent my_entity/my_project/SWEEP_ID --count 5
```

Esto ejecutará hasta 5 runs del sweep especificado. Cada run buscará en un conjunto de parámetros

3. **Script para Ejecutar varios agentes en paralelo:**

Automaticamente crea N agentes que en paralelo se dividen las ejecuciones dadas. 
En el siguinte ejemplo se establecen 100 ejecuciones del espacio de búsqueda, paralelizado en 5 agentes, para MLP + embedding periódico.
```python
python run_sweep.py --config configs/sweep_periodical_mlp.yaml --total-runs 100 --processes 2
```

