"""
R7 - Escenarios de benchmark.

El grafo usa el mismo contrato que algoritmos/a_star.py:
    adyacencia = {nodo: [(vecino, costo), ...]}
    coordenadas = {nodo: (latitud, longitud)}

Cada escenario recibe una COPIA del grafo para evitar que una prueba
contamine las siguientes. Cuando R3 esté disponible, esta capa permite
integrar su apply_scenario sin modificar el benchmark.
"""
from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple

SCENARIOS = (
    "trafico_normal",
    "trafico_alto",
    "calle_bloqueada",
    "sentido_unico",
    "obstaculos",
    "zona_incierta",
)

SCENARIO_LABELS = {
    "trafico_normal": "Tráfico normal",
    "trafico_alto": "Tráfico alto",
    "calle_bloqueada": "Calle bloqueada",
    "sentido_unico": "Sentido único",
    "obstaculos": "Obstáculos",
    "zona_incierta": "Zona incierta",
}


def _edge_count(graph):
    return sum(len(edges) for edges in graph.values())


def apply_scenario(adyacencia: dict, scenario: str, rng=None) -> dict:
    """
    Aplica un escenario sobre una copia del grafo y devuelve la copia.

    Nota de integración R3:
    cuando servicios.restricciones.apply_scenario esté implementado, se
    puede sustituir esta implementación provisional por el adaptador R3.
    Esta función mantiene el contrato local para que R7 sea ejecutable hoy.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Escenario desconocido: {scenario}")

    rng = rng or random.Random(42)
    graph = copy.deepcopy(adyacencia)

    edges = [(u, i, v, w) for u, outgoing in graph.items()
             for i, (v, w) in enumerate(outgoing)]
    if not edges:
        return graph

    if scenario == "trafico_normal":
        return graph

    if scenario == "trafico_alto":
        # Incrementa entre 35% y 90% el costo de varias vías.
        selected = rng.sample(edges, max(1, len(edges) // 2))
        for u, i, v, w in selected:
            graph[u][i] = (v, round(float(w) * rng.uniform(1.35, 1.90), 4))

    elif scenario == "calle_bloqueada":
        # Bloquea una conexión dirigida, representándola con costo infinito.
        u, i, v, w = rng.choice(edges)
        graph[u][i] = (v, float("inf"))

    elif scenario == "sentido_unico":
        # Elimina el sentido inverso de algunas calles bidireccionales.
        candidates = [(u, v) for u, _, v, _ in edges if u != v]
        rng.shuffle(candidates)
        removed = set()
        for u, v in candidates:
            if (u, v) in removed or (v, u) in removed:
                continue
            reverse = [(a, b) for a, outgoing in graph.items()
                       for b, _ in outgoing if a == v and b == u]
            if reverse:
                graph[u] = [(n, w) for n, w in graph[u] if n != v]
                removed.add((u, v))
                if len(removed) >= max(1, len(edges) // 8):
                    break

    elif scenario == "obstaculos":
        # Un obstáculo bloquea varias conexiones próximas (coste infinito).
        selected = rng.sample(edges, min(len(edges), max(2, len(edges) // 5)))
        for u, i, v, w in selected:
            graph[u][i] = (v, float("inf"))

    elif scenario == "zona_incierta":
        # Coste adicional moderado por incertidumbre de circulación.
        selected = rng.sample(edges, max(1, len(edges) // 3))
        for u, i, v, w in selected:
            graph[u][i] = (v, round(float(w) * rng.uniform(1.15, 1.55), 4))

    return graph


def get_scenarios():
    """Devuelve los identificadores y nombres legibles de los escenarios."""
    return [(key, SCENARIO_LABELS[key]) for key in SCENARIOS]
