import heapq
import math
import time
from dataclasses import dataclass


@dataclass
class ResultadoAStar:
    costo: float
    ruta: list
    explorados: int
    tiempo_ms: float
    encontrado: bool


def heuristica_haversine(coord_actual: tuple[float, float], coord_destino: tuple[float, float], velocidad_max_ms: float) -> float:
    """Estima el tiempo mínimo en segundos usando distancia en línea recta (Haversine) / velocidad máxima."""
    if velocidad_max_ms <= 0:
        return 0.0

    lat1, lon1 = coord_actual
    lat2, lon2 = coord_destino
    radio_tierra_m = 6371000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    distancia_m = radio_tierra_m * c

    return distancia_m / velocidad_max_ms


def a_star(
    adyacencia: dict,
    coordenadas: dict,
    inicio,
    destino,
    velocidad_max_ms: float = 10.0,
) -> ResultadoAStar:
    """
    Algoritmo A* para búsqueda de ruta óptima: f(n) = g(n) + h(n).

    - adyacencia: {nodo: [(vecino, costo), ...]}
    - coordenadas: {nodo: (lat, lon)}
    - inicio: nodo de partida
    - destino: nodo objetivo
    - velocidad_max_ms: velocidad máxima en m/s para mantener h(n) admisible
    """
    t0 = time.perf_counter()

    if inicio not in adyacencia or destino not in adyacencia:
        return ResultadoAStar(
            costo=math.inf,
            ruta=[],
            explorados=0,
            tiempo_ms=(time.perf_counter() - t0) * 1000.0,
            encontrado=False,
        )

    g_score = {inicio: 0.0}
    padre = {}
    cerrados = set()
    explorados = 0

    h0 = (
        heuristica_haversine(coordenadas[inicio], coordenadas[destino], velocidad_max_ms)
        if inicio in coordenadas and destino in coordenadas
        else 0.0
    )
    # Cola de prioridad ordenada por f(n) = g(n) + h(n)
    cola = [(h0, 0.0, inicio)]

    while cola:
        _, g_actual, actual = heapq.heappop(cola)

        if actual in cerrados:
            continue

        cerrados.add(actual)
        explorados += 1

        if actual == destino:
            ruta = []
            nodo_actual = destino
            while nodo_actual in padre:
                ruta.append(nodo_actual)
                nodo_actual = padre[nodo_actual]
            ruta.append(inicio)
            ruta.reverse()

            return ResultadoAStar(
                costo=g_actual,
                ruta=ruta,
                explorados=explorados,
                tiempo_ms=(time.perf_counter() - t0) * 1000.0,
                encontrado=True,
            )

        for vecino, peso in adyacencia.get(actual, []):
            if peso == math.inf or vecino in cerrados:
                continue

            tentative_g = g_actual + peso
            if tentative_g < g_score.get(vecino, math.inf):
                g_score[vecino] = tentative_g
                padre[vecino] = actual
                h_vecino = (
                    heuristica_haversine(coordenadas[vecino], coordenadas[destino], velocidad_max_ms)
                    if vecino in coordenadas and destino in coordenadas
                    else 0.0
                )
                f_vecino = tentative_g + h_vecino
                heapq.heappush(cola, (f_vecino, tentative_g, vecino))

    return ResultadoAStar(
        costo=math.inf,
        ruta=[],
        explorados=explorados,
        tiempo_ms=(time.perf_counter() - t0) * 1000.0,
        encontrado=False,
    )