class Node:

    def __init__(self, state, visited, cost, parent, color, island_size):
        self.visited = visited
        self.state = state
        self.color = color
        self.parent = parent
        self.island_size = island_size
        self.cost = cost
        self.value = 1000
        # Para los metodos de busqueda informados
        # si es a_search --> value = cost + heuristica
        # si es greedy ----> value = heuristica

    def set_cost(self, cost):
        self.cost = cost

    def set_value(self, value):
        self.value = value
