# Ejecutar Sweeps con Weights & Biases

Sigue estos pasos para lanzar y ejecutar un sweep en wandb:

1. **Crear el sweep:**
activa el entorno de conda y situate en la raiz del repositorio

```bash
wandb sweep configs/sweep.yaml
```
Este comando devolverá un sweep_id

2. **Ejecutar el agente para el sweep:**

Te devuelve el comando para ejecutar sweeps, algo del estilo wandb agent inaki/my-template/874plwyb

```bash
wandb agent my_entity/my_project/SWEEP_ID --count 5
```


Esto ejecutará hasta 5 runs del sweep especificado. Cada run buscará en un conjunto de parámetros