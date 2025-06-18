import argparse
import os
import re
import subprocess
import sys
from multiprocessing import Process


def usage_error(message):
    print(f"❌ {message}")
    sys.exit(1)


def run_agent(sweep_id, count, index):
    print(f"🚀 Iniciando agente {index + 1} con {count} runs...")
    subprocess.run(["wandb", "agent", sweep_id, "--count", str(count)], check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Lanzar múltiples agentes de W&B en paralelo desde un sweep.yaml"
    )
    parser.add_argument("--config", required=True, help="Ruta al archivo sweep.yaml")
    parser.add_argument(
        "--total-runs", required=True, type=int, help="Número total de ejecuciones"
    )
    parser.add_argument(
        "--processes", required=True, type=int, help="Número de procesos paralelos"
    )

    args = parser.parse_args()
    sweep_config = args.config
    total_runs = args.total_runs
    num_processes = args.processes

    # ========================
    # CONFIG W&B (EDITA ESTO)
    # ========================
    PROJECT = "sacyr"
    ENTITY = "inaki"

    if not os.path.isfile(sweep_config):
        usage_error(f"El archivo '{sweep_config}' no existe.")

    if total_runs <= 0 or num_processes <= 0:
        usage_error("--total-runs y --processes deben ser números positivos.")

    print(f"📤 Creando sweep desde {sweep_config}...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "wandb",
            "sweep",
            "--project",
            PROJECT,
            "--entity",
            ENTITY,
            sweep_config,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    output = result.stdout + result.stderr
    match = re.search(r"Creating sweep with ID: ([a-z0-9]+)", output)

    if not match:
        print(
            "❌ No se pudo extraer el sweep ID. Verifica que el sweep fue creado correctamente."
        )
        print(output)
        sys.exit(1)

    sweep_id = f"{ENTITY}/{PROJECT}/{match.group(1)}"
    print(f"✅ Sweep creado con ID: {sweep_id}")

    runs_per_process = total_runs // num_processes
    remainder = total_runs % num_processes

    print(f"🔁 Total runs: {total_runs}")
    print(f"🔄 Procesos: {num_processes}")
    print(f"⚙️  Runs por proceso: {runs_per_process}")

    processes = []

    for i in range(num_processes):
        extra = remainder if i == num_processes - 1 else 0
        count = runs_per_process + extra
        p = Process(target=run_agent, args=(sweep_id, count, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("✅ Todos los agentes han terminado.")


if __name__ == "__main__":
    main()
