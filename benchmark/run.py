"""
R7 - Ejecutor de benchmark.

Ejecución desde la raíz del proyecto:
    python -m benchmark.run

Genera 25 pares reproducibles (seed=42), los ejecuta en seis escenarios
y guarda una fila por combinación en results/benchmark.csv.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
import statistics
import time
import sys
from pathlib import Path

from benchmark.scenarios import SCENARIOS, SCENARIO_LABELS, apply_scenario

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CSV_PATH = RESULTS_DIR / "benchmark.csv"
SEED = 42
PAIR_COUNT = 25
MAX_SPEED_MS = 10.0


def _toy_graph():
    """Grafo de prueba pequeño, conectado y con coordenadas tipo Trujillo."""
    coordinates = {
        f"N{i}": (-8.1110 + (i // 4) * 0.0030, -79.0280 + (i % 4) * 0.0030)
        for i in range(12)
    }
    adjacency = {node: [] for node in coordinates}
    # Malla 3x4 con pesos en segundos (costos positivos).
    for r in range(3):
        for c in range(4):
            u = f"N{r*4+c}"
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 3 and 0 <= cc < 4:
                    v = f"N{rr*4+cc}"
                    # Pesos deliberadamente mayores que la heurística de juguete.
                    adjacency[u].append((v, 40.0 + ((r+c+rr+cc) % 4) * 5.0))
    # Algunas diagonales para que existan rutas alternativas.
    for r in range(2):
        for c in range(3):
            u, v = f"N{r*4+c}", f"N{(r+1)*4+c+1}"
            adjacency[u].append((v, 58.0))
            adjacency[v].append((u, 58.0))
    return adjacency, coordinates


def _load_real_map():
    """
    Intenta leer datos/mapa_trujillo.json. Se aceptan formatos habituales:
      {"adyacencia": {...}, "coordenadas": {...}}
      {"grafo": {"adyacencia": ..., "coordenadas": ...}}
    Si el JSON aún está vacío/no tiene ese esquema, usa el grafo de juguete.
    """
    path = ROOT / "datos" / "mapa_trujillo.json"
    if path.exists() and path.stat().st_size:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if "grafo" in data and isinstance(data["grafo"], dict):
                data = data["grafo"]
            adjacency = data.get("adyacencia") or data.get("adjacency")
            coordinates = data.get("coordenadas") or data.get("coordinates")
            if isinstance(adjacency, dict) and isinstance(coordinates, dict):
                normalized = {}
                for node, edges in adjacency.items():
                    normalized[node] = []
                    for edge in edges:
                        if isinstance(edge, dict):
                            neighbor = edge.get("vecino", edge.get("destino", edge.get("to")))
                            cost = edge.get("costo", edge.get("peso", edge.get("weight")))
                        else:
                            neighbor, cost = edge[0], edge[1]
                        normalized[node].append((neighbor, float(cost)))
                coords = {node: tuple(value) for node, value in coordinates.items()}
                if normalized and coords:
                    for node in normalized:
                        normalized.setdefault(node, [])
                    return normalized, coords, "mapa_trujillo.json"
        except (ValueError, TypeError, KeyError, IndexError):
            pass
    adjacency, coordinates = _toy_graph()
    return adjacency, coordinates, "grafo_de_juguete"


def _load_astar():
    """Carga el A* existente sin modificarlo; fallback solo si está vacío."""
    module_path = ROOT / "algoritmos" / "a_star.py"
    if module_path.exists() and module_path.stat().st_size:
        spec = importlib.util.spec_from_file_location("fireroute_a_star", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if hasattr(module, "a_star"):
            return module.a_star, "algoritmos/a_star.py"
    return _fallback_a_star, "fallback benchmark (hasta que A* esté disponible)"


def _fallback_a_star(adyacencia, coordenadas, inicio, destino,
                     velocidad_max_ms=MAX_SPEED_MS):
    """Fallback mínimo con el mismo resultado/atributos que el A* del proyecto."""
    start = time.perf_counter()
    if inicio not in adyacencia or destino not in adyacencia:
        return _Result(math.inf, [], 0, (time.perf_counter()-start)*1000, False)
    queue = [(0.0, inicio)]
    distance = {inicio: 0.0}
    parent = {}
    visited = set()
    explored = 0
    while queue:
        cost, node = __import__("heapq").heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        explored += 1
        if node == destino:
            path, current = [], destino
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(inicio)
            path.reverse()
            return _Result(cost, path, explored, (time.perf_counter()-start)*1000, True)
        for neighbor, weight in adyacencia.get(node, []):
            if not math.isfinite(float(weight)) or neighbor in visited:
                continue
            candidate = cost + float(weight)
            if candidate < distance.get(neighbor, math.inf):
                distance[neighbor] = candidate
                parent[neighbor] = node
                __import__("heapq").heappush(queue, (candidate, neighbor))
    return _Result(math.inf, [], explored, (time.perf_counter()-start)*1000, False)


class _Result:
    def __init__(self, costo, ruta, explorados, tiempo_ms, encontrado):
        self.costo = costo
        self.ruta = ruta
        self.explorados = explorados
        self.tiempo_ms = tiempo_ms
        self.encontrado = encontrado


def _generate_pairs(nodes, count=PAIR_COUNT, seed=SEED):
    """Pares distintos y reproducibles; origen nunca igual a destino."""
    rng = random.Random(seed)
    nodes = sorted(nodes, key=str)
    if len(nodes) < 2:
        raise ValueError("El grafo debe tener al menos dos nodos")
    pairs = set()
    max_pairs = len(nodes) * (len(nodes) - 1)
    target = min(count, max_pairs)
    while len(pairs) < target:
        origin, destination = rng.sample(nodes, 2)
        pairs.add((origin, destination))
    return sorted(pairs, key=lambda pair: (str(pair[0]), str(pair[1])))[:target]


def run_benchmark():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adjacency, coordinates, map_source = _load_real_map()
    astar, astar_source = _load_astar()
    pairs = _generate_pairs(adjacency.keys())

    columns = [
        "escenario_id", "escenario", "prueba", "origen", "destino",
        "ruta_encontrada", "costo", "nodos_explorados", "tiempo_ms",
        "longitud_ruta", "mapa_utilizado", "algoritmo_utilizado", "semilla"
    ]
    rows = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        # Mismo escenario base reproducible para cada conjunto de pares.
        scenario_graph = apply_scenario(
            adjacency, scenario, random.Random(SEED + scenario_index)
        )
        for test_number, (origin, destination) in enumerate(pairs, start=1):
            result = astar(
                scenario_graph, coordinates, origin, destination,
                velocidad_max_ms=MAX_SPEED_MS
            )
            rows.append({
                "escenario_id": scenario,
                "escenario": SCENARIO_LABELS[scenario],
                "prueba": test_number,
                "origen": origin,
                "destino": destination,
                "ruta_encontrada": bool(result.encontrado),
                "costo": result.costo if math.isfinite(result.costo) else "",
                "nodos_explorados": result.explorados,
                "tiempo_ms": round(float(result.tiempo_ms), 6),
                "longitud_ruta": len(result.ruta),
                "mapa_utilizado": map_source,
                "algoritmo_utilizado": astar_source,
                "semilla": SEED,
            })

    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    found = sum(row["ruta_encontrada"] for row in rows)
    times = [row["tiempo_ms"] for row in rows]
    print(f"Benchmark terminado: {len(rows)} ejecuciones")
    print(f"Mapa: {map_source}")
    print(f"Algoritmo: {astar_source}")
    print(f"Rutas encontradas: {found}/{len(rows)}")
    print(f"Tiempo promedio: {statistics.mean(times):.6f} ms")
    print(f"CSV: {CSV_PATH}")
    return rows


if __name__ == "__main__":
    run_benchmark()
