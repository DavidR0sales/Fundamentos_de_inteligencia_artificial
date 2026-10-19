"""
IMPORTANTE ES NECESARIO DESCARGAR NETWORKX

Modelado de problemas reales como problemas de busqueda en grafos.

Se modelan dos problemas:
  1. Ruta de reparto: un repartidor sale de un deposito, debe visitar
     varios clientes y busca minimizar los kilometros recorridos.
  2. Dosificacion de liquidos: dos recipientes de capacidad conocida
     deben medir una cantidad exacta de liquido minimizando los litros
     manipulados.

Para cada problema se define:
  - Estado:  descripcion completa de la situacion en un instante.
  - Acciones: operaciones permitidas en cada estado.
  - Costo:   valor positivo asociado a cada accion.
  - Meta:    condicion que debe cumplir el estado final.

El generador de estados (generar_grafo) explora el espacio de forma
sistematica y construye un grafo dirigido y ponderado con NetworkX,
donde cada nodo es un estado y cada arista es una accion con su costo.

Ejecucion:
    python modelado_busqueda.py
Requisitos:
    pip install networkx matplotlib
"""

import time
from collections import deque

import matplotlib

matplotlib.use("Agg")  # Backend sin ventana: guarda las imagenes en archivo
import matplotlib.pyplot as plt
import networkx as nx


# ---------------------------------------------------------------------------
# Interfaz comun para cualquier problema de busqueda
# ---------------------------------------------------------------------------
class ProblemaBusqueda:
    """Clase base que define los cuatro elementos de un problema de busqueda."""

    def estado_inicial(self):
        """Devuelve el estado desde el cual inicia la busqueda."""
        raise NotImplementedError

    def acciones(self, estado):
        """
        Devuelve una lista de tuplas (nombre_accion, estado_resultante, costo)
        con todas las acciones validas aplicables a 'estado'.
        """
        raise NotImplementedError

    def es_meta(self, estado):
        """Devuelve True si 'estado' satisface la condicion de meta."""
        raise NotImplementedError

    def etiqueta(self, estado):
        """Texto legible del estado, usado en los diagramas."""
        return str(estado)


# ---------------------------------------------------------------------------
# Problema 1: ruta de reparto
# ---------------------------------------------------------------------------
class RutaReparto(ProblemaBusqueda):
    """
    Estado:   (ubicacion, entregados) donde 'entregados' es un frozenset con
              los clientes que ya recibieron su paquete.
    Acciones: desplazarse por una carretera hacia una localidad vecina. Si la
              localidad destino es un cliente, la entrega se realiza al llegar.
    Costo:    kilometros de la carretera recorrida.
    Meta:     todos los clientes atendidos (no se exige regresar al deposito).

    Los kilometros son valores ilustrativos para el ejercicio.
    """

    # Carreteras dirigidas: (origen, destino, km). La de Chalco hacia
    # Los Reyes es de un solo sentido para que el grafo sea realmente dirigido.
    CARRETERAS = [
        ("Ixtapaluca", "Chalco", 14),
        ("Chalco", "Ixtapaluca", 14),
        ("Ixtapaluca", "Los Reyes", 10),
        ("Los Reyes", "Ixtapaluca", 10),
        ("Los Reyes", "Texcoco", 18),
        ("Texcoco", "Los Reyes", 18),
        ("Ixtapaluca", "Texcoco", 28),
        ("Texcoco", "Ixtapaluca", 30),
        ("Chalco", "Los Reyes", 17),
    ]

    def __init__(self, deposito="Ixtapaluca",
                 clientes=("Chalco", "Los Reyes", "Texcoco")):
        self.deposito = deposito
        self.clientes = frozenset(clientes)
        # Lista de adyacencia construida a partir de las carreteras
        self.vecinos = {}
        for origen, destino, km in self.CARRETERAS:
            self.vecinos.setdefault(origen, []).append((destino, km))

    def estado_inicial(self):
        # Si el deposito fuera tambien cliente, se marca como atendido
        entregados = frozenset([self.deposito]) & self.clientes
        return (self.deposito, entregados)

    def acciones(self, estado):
        ubicacion, entregados = estado
        resultado = []
        for destino, km in self.vecinos.get(ubicacion, []):
            # Al llegar a un cliente se registra la entrega automaticamente
            nuevos = entregados | ({destino} & self.clientes)
            resultado.append((f"ir a {destino}", (destino, frozenset(nuevos)), km))
        return resultado

    def es_meta(self, estado):
        return estado[1] == self.clientes

    def etiqueta(self, estado):
        ubicacion, entregados = estado
        faltan = len(self.clientes - entregados)
        return f"{ubicacion}\nentregas: {len(entregados)}/{len(self.clientes)}\nfaltan: {faltan}"


# ---------------------------------------------------------------------------
# Problema 2: dosificacion de liquidos con dos recipientes
# ---------------------------------------------------------------------------
class DosificacionLiquidos(ProblemaBusqueda):
    """
    Estado:   (a, b) litros contenidos en el recipiente A y en el B.
    Acciones: llenar A, llenar B, vaciar A, vaciar B, verter A en B,
              verter B en A. Se descartan las acciones que no cambian el estado.
    Costo:    litros de liquido que se agregan, se desechan o se trasvasan
              en la accion (siempre mayor que cero).
    Meta:     que el recipiente B contenga exactamente 'objetivo' litros.
    """

    def __init__(self, cap_a=3, cap_b=5, objetivo=4):
        self.cap_a = cap_a
        self.cap_b = cap_b
        self.objetivo = objetivo

    def estado_inicial(self):
        return (0, 0)

    def acciones(self, estado):
        a, b = estado
        candidatas = []

        # Llenar un recipiente: costo igual a lo que falta para llenarlo
        candidatas.append(("llenar A", (self.cap_a, b), self.cap_a - a))
        candidatas.append(("llenar B", (a, self.cap_b), self.cap_b - b))

        # Vaciar un recipiente: costo igual a lo que se desecha
        candidatas.append(("vaciar A", (0, b), a))
        candidatas.append(("vaciar B", (a, 0), b))

        # Verter: se mueve lo que quepa en el destino o lo que haya en origen
        mover_ab = min(a, self.cap_b - b)
        candidatas.append(("verter A en B", (a - mover_ab, b + mover_ab), mover_ab))
        mover_ba = min(b, self.cap_a - a)
        candidatas.append(("verter B en A", (a + mover_ba, b - mover_ba), mover_ba))

        # Se conservan solo las acciones con efecto real (costo positivo)
        return [(n, s, c) for n, s, c in candidatas if c > 0]

    def es_meta(self, estado):
        return estado[1] == self.objetivo

    def etiqueta(self, estado):
        return f"({estado[0]},{estado[1]})"


# ---------------------------------------------------------------------------
# Generador de estados y construccion del grafo con NetworkX
# ---------------------------------------------------------------------------
def generar_grafo(problema):
    """
    Genera el espacio de estados completo mediante una exploracion en
    anchura y lo almacena en un grafo dirigido y ponderado.

    Cada nodo guarda el atributo 'meta' (bool). Cada arista guarda
    'weight' (costo) y 'accion' (nombre de la accion).

    Complejidad: O(V * b) en tiempo, con V estados alcanzables y b factor
    de ramificacion; O(V + E) en espacio.
    """
    grafo = nx.DiGraph()
    inicial = problema.estado_inicial()
    grafo.add_node(inicial, meta=problema.es_meta(inicial))

    cola = deque([inicial])
    visitados = {inicial}  # Evita expandir dos veces el mismo estado

    while cola:
        actual = cola.popleft()
        for accion, siguiente, costo in problema.acciones(actual):
            if siguiente not in visitados:
                visitados.add(siguiente)
                grafo.add_node(siguiente, meta=problema.es_meta(siguiente))
                cola.append(siguiente)
            # Si dos acciones llevan al mismo estado se conserva la mas barata
            if grafo.has_edge(actual, siguiente):
                if costo >= grafo[actual][siguiente]["weight"]:
                    continue
            grafo.add_edge(actual, siguiente, weight=costo, accion=accion)
    return grafo, inicial


# ---------------------------------------------------------------------------
# Algoritmos de busqueda sobre el grafo generado
# ---------------------------------------------------------------------------
def busqueda_anchura(grafo, inicial):
    """
    Busqueda en anchura (BFS): encuentra la solucion con menos acciones,
    sin considerar los costos. Devuelve (camino, nodos_expandidos).
    """
    cola = deque([inicial])
    padre = {inicial: None}
    expandidos = 0
    while cola:
        actual = cola.popleft()
        expandidos += 1
        if grafo.nodes[actual]["meta"]:
            return reconstruir_camino(padre, actual), expandidos
        for vecino in grafo.successors(actual):
            if vecino not in padre:
                padre[vecino] = actual
                cola.append(vecino)
    return None, expandidos


def costo_uniforme(grafo, inicial):
    """
    Busqueda de costo uniforme (Dijkstra con cola de prioridad): encuentra
    la solucion de menor costo total. Devuelve (camino, costo, expandidos).
    """
    import heapq

    contador = 0  # Desempate estable para estados no comparables entre si
    frontera = [(0, contador, inicial)]
    mejor = {inicial: 0}
    padre = {inicial: None}
    cerrados = set()
    expandidos = 0
    while frontera:
        costo, _, actual = heapq.heappop(frontera)
        if actual in cerrados:
            continue
        cerrados.add(actual)
        expandidos += 1
        if grafo.nodes[actual]["meta"]:
            return reconstruir_camino(padre, actual), costo, expandidos
        for vecino in grafo.successors(actual):
            nuevo = costo + grafo[actual][vecino]["weight"]
            if nuevo < mejor.get(vecino, float("inf")):
                mejor[vecino] = nuevo
                padre[vecino] = actual
                contador += 1
                heapq.heappush(frontera, (nuevo, contador, vecino))
    return None, float("inf"), expandidos


def reconstruir_camino(padre, destino):
    """Recorre el diccionario de padres hacia atras para armar el camino."""
    camino = []
    while destino is not None:
        camino.append(destino)
        destino = padre[destino]
    return camino[::-1]


def verificar_con_networkx(grafo, inicial):
    """
    Contrasta el costo uniforme propio con el algoritmo de Dijkstra de
    NetworkX, tomando la meta alcanzable de menor distancia.
    """
    distancias = nx.single_source_dijkstra_path_length(grafo, inicial)
    metas = [n for n in grafo.nodes if grafo.nodes[n]["meta"] and n in distancias]
    return min(distancias[n] for n in metas) if metas else float("inf")


# ---------------------------------------------------------------------------
# Dibujo de los diagramas
# ---------------------------------------------------------------------------
def posiciones_por_niveles(grafo, inicial):
    """
    Distribuye los nodos en columnas segun su distancia (en acciones)
    desde el estado inicial, lo que deja el grafo legible de izquierda
    a derecha.
    """
    niveles = nx.single_source_shortest_path_length(grafo, inicial)
    columnas = {}
    for nodo, nivel in niveles.items():
        columnas.setdefault(nivel, []).append(nodo)
    pos = {}
    for nivel, nodos in columnas.items():
        for i, nodo in enumerate(nodos):
            pos[nodo] = (nivel * 3.0, -(i - (len(nodos) - 1) / 2) * 1.6)
    return pos


def dibujar_grafo(grafo, inicial, camino, problema, titulo, archivo,
                  pos=None, etiquetas_aristas=True, tam=(16, 9)):
    """
    Dibuja el grafo y lo guarda como imagen PNG.
      Verde: estado inicial. Dorado: estados meta.
      Rojo:  camino optimo por costo uniforme.
    """
    if pos is None:
        pos = posiciones_por_niveles(grafo, inicial)

    fig, ax = plt.subplots(figsize=tam)

    colores = []
    for nodo in grafo.nodes:
        if nodo == inicial:
            colores.append("#7bc47f")
        elif grafo.nodes[nodo]["meta"]:
            colores.append("#f2c14e")
        else:
            colores.append("#cfd8e3")

    aristas_camino = set(zip(camino, camino[1:])) if camino else set()
    aristas_normales = [e for e in grafo.edges if e not in aristas_camino]

    nx.draw_networkx_nodes(grafo, pos, node_color=colores, node_size=2300,
                           edgecolors="#333333", ax=ax)
    nx.draw_networkx_labels(grafo, pos,
                            {n: problema.etiqueta(n) for n in grafo.nodes},
                            font_size=7, ax=ax)
    nx.draw_networkx_edges(grafo, pos, edgelist=aristas_normales,
                           edge_color="#888888", arrows=True, arrowsize=14,
                           node_size=2300, connectionstyle="arc3,rad=0.12",
                           width=1.0, ax=ax)
    nx.draw_networkx_edges(grafo, pos, edgelist=list(aristas_camino),
                           edge_color="#d62728", arrows=True, arrowsize=18,
                           node_size=2300, connectionstyle="arc3,rad=0.12",
                           width=3.0, ax=ax)

    if etiquetas_aristas:
        etiquetas = nx.get_edge_attributes(grafo, "weight")
        nx.draw_networkx_edge_labels(grafo, pos, etiquetas, font_size=7,
                                     label_pos=0.3, rotate=False,
                                     connectionstyle="arc3,rad=0.12", ax=ax)

    ax.set_title(titulo, fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ejecucion de un problema completo
# ---------------------------------------------------------------------------
def resolver(nombre, problema, archivo_png, pos_fn=None, etiquetas_aristas=True):
    """Genera el grafo, ejecuta ambas busquedas, imprime metricas y dibuja."""
    print("=" * 70)
    print(nombre)
    print("=" * 70)

    t0 = time.perf_counter()
    grafo, inicial = generar_grafo(problema)
    t_gen = time.perf_counter() - t0

    n_metas = sum(1 for n in grafo.nodes if grafo.nodes[n]["meta"])
    grado_medio = grafo.number_of_edges() / grafo.number_of_nodes()
    print(f"Estados alcanzables (V): {grafo.number_of_nodes()}")
    print(f"Transiciones (E):        {grafo.number_of_edges()}")
    print(f"Estados meta:            {n_metas}")
    print(f"Factor de ramificacion medio: {grado_medio:.2f}")
    print(f"Tiempo de generacion:    {t_gen * 1000:.3f} ms")

    camino_bfs, exp_bfs = busqueda_anchura(grafo, inicial)
    camino_ucs, costo_ucs, exp_ucs = costo_uniforme(grafo, inicial)
    costo_nx = verificar_con_networkx(grafo, inicial)

    costo_bfs = sum(grafo[u][v]["weight"] for u, v in zip(camino_bfs, camino_bfs[1:]))
    print(f"\nBFS:  {len(camino_bfs) - 1} acciones, costo {costo_bfs}, "
          f"nodos expandidos {exp_bfs}")
    print(f"UCS:  {len(camino_ucs) - 1} acciones, costo {costo_ucs}, "
          f"nodos expandidos {exp_ucs}")
    print(f"Verificacion con Dijkstra de NetworkX: costo minimo {costo_nx}")
    assert costo_ucs == costo_nx, "El costo uniforme no coincide con NetworkX"

    print("\nPlan de costo minimo:")
    for u, v in zip(camino_ucs, camino_ucs[1:]):
        datos = grafo[u][v]
        print(f"  {problema.etiqueta(u).replace(chr(10), ' ')}  --{datos['accion']} "
              f"(costo {datos['weight']})-->  "
              f"{problema.etiqueta(v).replace(chr(10), ' ')}")

    pos = pos_fn(grafo) if pos_fn else None
    dibujar_grafo(grafo, inicial, camino_ucs, problema,
                  f"{nombre} - grafo del espacio de estados",
                  archivo_png, pos=pos, etiquetas_aristas=etiquetas_aristas)
    print(f"\nDiagrama guardado en: {archivo_png}\n")


def main():
    # Problema 1: ruta de reparto
    resolver("Problema 1: Ruta de reparto",
             RutaReparto(),
             "grafo_ruta_reparto.png")

    # Problema 2: dosificacion de liquidos. Los nodos se ubican en una
    # cuadricula usando directamente los litros (a, b) como coordenadas.
    resolver("Problema 2: Dosificacion de liquidos",
             DosificacionLiquidos(cap_a=3, cap_b=5, objetivo=4),
             "grafo_dosificacion.png",
             pos_fn=lambda g: {n: (n[0] * 3.0, n[1] * 1.8) for n in g.nodes},
             etiquetas_aristas=False)


if __name__ == "__main__":
    main()
