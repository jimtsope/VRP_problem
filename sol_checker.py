class Node:
    def __init__(self, node_id, family, costs, demand):
        self.id = node_id
        self.family = family
        self.costs = costs
        self.demand = demand
        self.is_depot = False
        self.is_routed = False
        self.route = None


class Family:
    def __init__(self, family_id, nodes, demand, required):
        self.id = family_id
        self.nodes = nodes
        self.demand = demand
        self.required = required


class Route:
    def __init__(self, route_id, sequence_of_nodes, capacity, cost, load):
        self.id = route_id
        self.sequence_of_nodes = sequence_of_nodes
        self.capacity = capacity
        self.cost = cost
        self.load = load

class Solution:
    def __init__(self):
        self.routes = []
        self.cost = 0.0


class Model:
    def __init__(self):
        self.num_nodes = 0
        self.num_fam = 0
        self.num_req = 0
        self.capacity = 0
        self.vehicles = 0
        self.fam_members = []
        self.fam_req = []
        self.fam_demand = []
        self.cost_matrix = []
        self.depot = None
        self.nodes = []
        self.customers = []
        self.families = []

def check_feasibility(model, sol):
    return (check_cost(model, sol) and
            check_family_demands(model, sol) and
            check_route_capacity(model, sol) and
            check_customer_uniqueness(sol) and
            check_vehicles(model, sol))


def check_family_demands(model, sol):
    visited_per_family = [0] * model.num_fam
    for rt in sol.routes:
        for i in range(1, len(rt.sequence_of_nodes) - 1):
            visited_per_family[rt.sequence_of_nodes[i].family.id] += 1

    for i, visited_count in enumerate(visited_per_family):
        if visited_count != model.fam_req[i]:
            print(f"Family demand violation for family: {i}")
            return False
    return True


def check_route_capacity(model, sol):
    for rt in sol.routes:
        rt_load = sum(n.demand for n in rt.sequence_of_nodes)
        if rt_load > model.capacity:
            print(f"Capacity violation in route: {rt.id}")
            return False
    return True


def check_customer_uniqueness(sol):
    customers = set()
    for rt in sol.routes:
        for i in range(1, len(rt.sequence_of_nodes) - 1):
            customer = rt.sequence_of_nodes[i]
            if customer in customers:
                print(f"Customer uniqueness violation for customer: {customer.id}")
                return False
            customers.add(customer)
    return True


def check_vehicles(model, sol):
    if len(sol.routes) != model.vehicles:
        return False
    return True


def check_cost(model, sol):
    total_cost = 0.0
    for rt in sol.routes:
        rt_cost = 0.0
        for i in range(len(rt.sequence_of_nodes) - 1):
            rt_cost += model.cost_matrix[rt.sequence_of_nodes[i].id][rt.sequence_of_nodes[i + 1].id]
        if abs(rt.cost - rt_cost) > 1e-6:
            print(f"Cost inconsistency in route: {rt.id}")
            return False
        total_cost += rt.cost
    if abs(sol.cost - total_cost) > 1e-6:
        print("Total solution cost inconsistency")
        return False
    return True

def find_node_by_id(model, node_id):
    for node in model.nodes:
        if node.id == node_id:
            return node
    return None

def load_solution(model, file_name):
    solution = Solution()

    with open(file_name, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print(f"Warning: Solution file '{file_name}' is empty.")
        return solution
    try:
        cost_str = lines[0].split(':')[1].strip()
        solution.cost = float(cost_str)
    except (IndexError, ValueError) as e:
        print(f"Error parsing cost from file '{file_name}': {e}")
        return solution

    route_id_counter = 0
    for line in lines[1:]:
        node_ids_str = line.split('-')
        sequence_of_nodes = []
        for node_id_str in node_ids_str:
            node_id = int(node_id_str)
            node = find_node_by_id(model, node_id)
            if node:
                sequence_of_nodes.append(node)
            else:
                print(f"Error: Node with ID {node_id} not found in model.")
                return Solution()

        if sequence_of_nodes:
            route = Route(route_id=route_id_counter, sequence_of_nodes=sequence_of_nodes, capacity=model.capacity,
                          cost=0.0, load=0.0)
            calculate_route_cost_demand(route, model.cost_matrix)
            solution.routes.append(route)
            route_id_counter += 1

    while len(solution.routes) < model.vehicles:
        route = Route(route_id=route_id_counter, sequence_of_nodes=[model.depot, model.depot], capacity=model.capacity,
                      cost=0.0, load=0.0)
        solution.routes.append(route)
        route_id_counter += 1

    return solution


def calculate_route_cost_demand(route, cost_matrix):
    for i in range(len(route.sequence_of_nodes) - 1):
        node_a = route.sequence_of_nodes[i]
        node_b = route.sequence_of_nodes[i + 1]
        route.cost += cost_matrix[node_a.id][node_b.id]
        route.load += node_a.demand

def create_model(file_path):
    model = Model()
    try:
        with open(file_path, 'r') as sr:
            # 1st line
            line = sr.readline()
            parts = line.split()
            model.num_nodes = int(parts[0])
            model.num_fam = int(parts[1])
            model.num_req = int(parts[2])
            model.capacity = int(parts[3])
            model.vehicles = int(parts[4])

            # 2nd line
            line = sr.readline()
            parts = [p for p in line.split() if p]
            model.fam_members = [int(part) for part in parts]

            # 3rd line
            line = sr.readline()
            parts = [p for p in line.split() if p]
            model.fam_req = [int(part) for part in parts]

            # 4th line
            line = sr.readline()
            parts = [p for p in line.split() if p]
            model.fam_demand = [int(part) for part in parts]

            # 5th line onwards (cost matrix)
            cost_matrix = []
            for line in sr:
                if not line.strip():
                    continue
                node_costs = []
                parts = [p for p in line.split() if p]
                for part in parts:
                    cost = int(part)
                    if cost < 0:
                        node_costs.append(10000)
                    else:
                        node_costs.append(cost)
                cost_matrix.append(node_costs)
            model.cost_matrix = cost_matrix

        return _create_nodes_families(model)

    except Exception as e:
        print(f"Exception: {e}")
        return Model()


def _create_nodes_families(model):
    families = []
    nodes = []
    customers = []

    # Family initialization
    for i in range(model.num_fam):
        family = Family(i, [], model.fam_demand[i], model.fam_req[i])
        families.append(family)

    # Depot initialization
    depot = Node(0, None, model.cost_matrix[0], 0)
    depot.is_depot = True
    nodes.append(depot)
    model.depot = depot

    # Nodes and customers initialization
    for i in range(1, len(model.cost_matrix)):
        fam_index = _find_node_family(model, i)
        node = Node(i, families[fam_index], model.cost_matrix[i], families[fam_index].demand)
        nodes.append(node)
        customers.append(node)

    # Add customer nodes to families
    for customer in customers:
        families[customer.family.id].nodes.append(customer)

    model.families = families
    model.nodes = nodes
    model.customers = customers

    return model


def _find_node_family(model, node_id):
    c = 0
    prev = 0
    for i in model.fam_members:
        if node_id <= i + prev:
            return c
        else:
            prev = prev + i
            c += 1
    return c


if __name__ == '__main__':
    model = create_model('instance.txt')
    sol = load_solution(model, 'Final_Solution.txt')
    if check_feasibility(model, sol):
        print(f"Valid solution. Cost: {sol.cost}")
    else:
        print("Invalid solution.")