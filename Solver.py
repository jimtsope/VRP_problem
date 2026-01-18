
#  1) build_solution
#  2) choose_nodes_per_family_grasp  (inside GRASP)
#  3) clarke_wright_construction     (inside GRASP)
#  4) local_search                   (used in GRASP + ILS)
#  5) local search moves:
#       - two_opt_route
#       - relocate_move
#       - swap_move
#       - or_opt_move
#       - two_opt_star
#       - family_node_replacement
#  6) Iterated Local Search:
#       - intensify_with_ils
#       - random_kick
#       - clone_routes
#  7) Update_route_cost_and_load
#  8) save_solution

import random
from VRPModel import Route, Solution, load_model, calculate_route_cost_demand


# Model of 6 Parameters

SELECTED_SEED   = 4      # fixed seed  , 4
MAX_GRASP_ITERS = 500    # how many GRASP constructions  , 500
LS_MAX_PASSES   = 13    # how deep local search can go , 13
ILS_ITERS       = 2700  # how many ILS iterations, 2700
KICK_MOVES      = 15     # relocations per ILS kick, 15
flag            = True  # as a print flag

# 1. Main build of our solution
def build_solution(seed: int = SELECTED_SEED):

#   Strategy:
#    a. GRASP multi-start:
#         - choose nodes per family (GRASP)
#         - build initial routes (Clarke and Wright)
#         - repair if too many routes
#         - local_search
#         - keep best solution
#    b.  Iterated Local Search:
#        - from best GRASP solution
#        - repeat: kick + local_search
#        - keep best solution
#    c. Pad routes so we have exactly 'model.vehicles' routes
#    d. Return (model, best_solution)

    # Create random number generator object
    rnd = random.Random(seed)

    # Load problem instance
    model = load_model()

    best_solution = None
    best_cost = 10**10

    # a. GRASP multi-start
    for i in range(MAX_GRASP_ITERS):

        # choose nodes per family using GRASP
        chosen_customers = choose_nodes_per_family_grasp(model, rnd)

        # build initial routes with Clarke and Wright
        routes = clarke_wright_construction(model, chosen_customers, rnd)

        # if we produced more routes than available vehicles, repair
        if len(routes) > model.vehicles:
            # sort routes by total load (heaviest first)
            routes.sort(
                key=lambda r: sum(n.demand for n in r.sequence_of_nodes),
                reverse=True)

            # keep only as many routes as we have vehicles
            kept_routes = routes[:model.vehicles]
            extra_routes = routes[model.vehicles:]

            # collect all customers from the extra routes
            remaining_nodes = []
            for extra_route in extra_routes:
                # skip depot at start and end
                for node in extra_route.sequence_of_nodes[1:-1]:
                    remaining_nodes.append(node)

            routes = kept_routes
            C = model.cost_matrix

            # insert each remaining customer in the best place of the kept routes
            for node in remaining_nodes:
                best_increase = None
                best_pos = None
                best_route_choice = None

                for r in routes:
                    # current load of route r
                    load_r = sum(n.demand for n in r.sequence_of_nodes)
                    if load_r + node.demand > model.capacity:
                        continue

                    seq = r.sequence_of_nodes
                    # try inserting between each pair (a,b)
                    for i in range(len(seq) - 1):
                        a = seq[i]
                        b = seq[i + 1]
                        increase = (
                            C[a.id][node.id]
                            + C[node.id][b.id]
                            - C[a.id][b.id]
                        )
                        if best_increase is None or increase < best_increase:
                            best_increase = increase
                            best_pos = i
                            best_route_choice = r

                if best_route_choice is None:
                    # insert into the lightest route in position 1
                    routes.sort(
                        key=lambda r: sum(n.demand for n in r.sequence_of_nodes)
                    )
                    r = routes[0]
                    r.sequence_of_nodes.insert(1, node)
                else:
                    best_route_choice.sequence_of_nodes.insert(best_pos + 1, node)

            # recompute cost and load for all routes
            for r in routes:
                update_route_cost_and_load(r, model.cost_matrix)

        # improve this solution with local search
        sol = local_search(model, routes, max_passes=LS_MAX_PASSES)
        if flag:
            print(f"GRASP current cost : {sol.cost}")

        # keep best GRASP+LS solution
        if best_solution is None or sol.cost < best_cost:
            best_cost = sol.cost
            best_solution = sol
            if flag:
                print(f"GRASP best: {best_cost}")


    # b. Iterated Local Search
    best_solution = intensify_with_ils(model, best_solution, rnd, ils_iters=ILS_ITERS)

    # c. ensure we have exactly model.vehicles routes (add empty ones if needed)
    while len(best_solution.routes) < model.vehicles:
        empty_route = Route(
            route_id=len(best_solution.routes),
            sequence_of_nodes=[model.depot, model.depot],
            capacity=model.capacity,
            cost=0,
            load=0,
        )
        best_solution.routes.append(empty_route)

    # recompute total cost (sum of route costs)
    total_cost = 0
    for r in best_solution.routes:
        total_cost += r.cost
    best_solution.cost = total_cost
    # d. return the final solution
    return model, best_solution


# 2. GRASP: Family node selection
def choose_nodes_per_family_grasp(model, rnd, alpha=0.2, cand_factor=3.0):

#    For each family we must choose exactly 'family.required' nodes.
#      - sort family nodes by distance to depot
#      - keep top K as candidate list
#      - for each required node:
#          - compute a score based on distance to depot and distance to already selected nodes
#          - build a restricted candidate list using alpha (greediness vs randomness)
#          - pick randomly from this list

    C = model.cost_matrix
    depot_id = model.depot.id

    all_selected_nodes = []

    for family in model.families:
        family_nodes = family.nodes
        required_count = family.required

        # sort nodes by distance from depot (closest first)
        nodes_sorted = sorted(
            family_nodes,
            key=lambda node: C[depot_id][node.id]
        )

        # choose candidate list size
        base_K = required_count + 2
        scaled_K = int(cand_factor * required_count)
        K = max(base_K, scaled_K)
        if K > len(nodes_sorted):
            K = len(nodes_sorted)

        candidates = nodes_sorted[:K]
        selected = []
        available = list(candidates)

        def node_score(node):
            #Score used for GRASP:
            #  if nothing selected:
            #    score = distance depot to node
            #  else:
            #    score = 0.5*(distance depot to node)
            #            + 0.5*(distance node to the closest selected node)

            if len(selected) == 0:
                return C[depot_id][node.id]

            dist_to_depot = C[depot_id][node.id]
            nearest_sel_dist = None
            for sel in selected:
                d = C[node.id][sel.id]
                if nearest_sel_dist is None or d < nearest_sel_dist:
                    nearest_sel_dist = d

            return 0.5 * dist_to_depot + 0.5 * nearest_sel_dist

        # pick exactly family.required nodes
        for l in range(required_count):
            scored = []
            for node in available:
                s_val = node_score(node)
                scored.append((s_val, node))

            scored.sort(key=lambda pair: pair[0])
            best = scored[0][0]
            worst = scored[-1][0]

            if worst > best:
                threshold = best + alpha * (worst - best)
            else:
                threshold = best

            # build RCL
            rcl = []
            for s_val, node in scored:
                if s_val <= threshold:
                    rcl.append(node)

            if len(rcl) == 0:
                # degenerate case
                rcl = [node for (s_val, node) in scored]

            chosen_node = rnd.choice(rcl)
            selected.append(chosen_node)
            available.remove(chosen_node)

        # add this family's selected nodes to global list
        for node in selected:
            all_selected_nodes.append(node)

    return all_selected_nodes



# 3. GRASP: CLARKE and WRIGHT construction solution

def clarke_wright_construction(model, customers, rnd):
#    Build an initial solution using Clarke and Wright algorithm.
#    1) Start with one route per customer: depot to customer to depot.
#    2) Compute savings s(i,j) = c(0,i) + c(0,j) - c(i,j).
#    3) Sort savings descending, try to merge routes for each pair.

    C = model.cost_matrix
    depot = model.depot

    # STEP 1: one route per customer
    routes = []
    for idx, customer in enumerate(customers):
        seq = [depot, customer, depot]
        r = Route(
            route_id=idx,
            sequence_of_nodes=seq,
            capacity=model.capacity,
            cost=0,
            load=0,
        )
        update_route_cost_and_load(r, C)
        routes.append(r)

    def get_route_load(route):
        total = 0
        for node in route.sequence_of_nodes:
            total += node.demand
        return total

    def rebuild_node_to_route():

    #    Map node_id to route containing that node.
    #    (ignores depot nodes)

        m = {}
        for r in routes:
            for node in r.sequence_of_nodes[1:-1]:
                m[node.id] = r
        return m

    node_to_route = rebuild_node_to_route()

    # STEP 2: compute savings
    savings = []
    n = len(customers)
    for i_idx in range(n):
        ni = customers[i_idx]
        for j_idx in range(i_idx + 1, n):
            nj = customers[j_idx]
            s_val = (
                C[depot.id][ni.id]
                + C[depot.id][nj.id]
                - C[ni.id][nj.id]
            )
            savings.append((s_val, ni.id, nj.id))

    rnd.shuffle(savings)
    savings.sort(reverse=True, key=lambda triple: triple[0])

    # STEP 3: merge routes using savings if conditions are satisfied:
    # - i, j endpoint
    # - i, j belong to different routes
    # - respect constraints capacity
    for s_val, i_id, j_id in savings:
        ri = node_to_route.get(i_id)
        rj = node_to_route.get(j_id)

        if ri is None or rj is None:
            continue
        if ri is rj:
            continue

        ri_first = ri.sequence_of_nodes[1]
        ri_last = ri.sequence_of_nodes[-2]
        rj_first = rj.sequence_of_nodes[1]
        rj_last = rj.sequence_of_nodes[-2]

        merged = None


        if ri_last.id == i_id and rj_first.id == j_id:
            new_seq = ri.sequence_of_nodes[:-1] + rj.sequence_of_nodes[1:]
            merged = Route(-1, new_seq, model.capacity, 0, 0)

        elif ri_first.id == i_id and rj_last.id == j_id:
            mid_ri = list(ri.sequence_of_nodes[1:-1])
            mid_ri.reverse()
            mid_rj = list(rj.sequence_of_nodes[1:-1])
            mid_rj.reverse()
            new_seq = [depot] + mid_rj + mid_ri + [depot]
            merged = Route(-1, new_seq, model.capacity, 0, 0)

        elif ri_last.id == i_id and rj_last.id == j_id:
            mid_rj = list(rj.sequence_of_nodes[1:-1])
            mid_rj.reverse()
            new_seq = ri.sequence_of_nodes[:-1] + mid_rj + [depot]
            merged = Route(-1, new_seq, model.capacity, 0, 0)

        elif ri_first.id == i_id and rj_first.id == j_id:
            mid_ri = list(ri.sequence_of_nodes[1:-1])
            mid_ri.reverse()
            new_seq = [depot] + mid_ri + rj.sequence_of_nodes[1:]
            merged = Route(-1, new_seq, model.capacity, 0, 0)

        if merged is None:
            continue

        if get_route_load(merged) > model.capacity:
            continue

        update_route_cost_and_load(merged, C)
        routes.remove(ri)
        routes.remove(rj)
        routes.append(merged)
        node_to_route = rebuild_node_to_route()

    # renumber routes and recompute cost/load
    for rid, r in enumerate(routes):
        r.id = rid
        update_route_cost_and_load(r, C)

    return routes



# 4) Local Search Strategy

def local_search(model, routes, max_passes: int = LS_MAX_PASSES):

#    Apply all local moves repeatedly until:
#      - no move improves the solution or
#      - we reach 'max_passes' passes.
#   Moves inside each pass:
#      - 2-opt (two_opt_route)
#      - relocate (relocate_move)
#      - swap (swap_move)
#      - Or-opt (or_opt_move)
#      - two_opt_star
#      - family node replacement (family_node_replacement)

    passes_done = 0

    while passes_done < max_passes:
        passes_done += 1
        improved = False

        # 2-opt on each route
        for r in routes:
            if two_opt_route(r, model.cost_matrix):
                improved = True

        # cross-route moves
        if relocate_move(model, routes):
            improved = True

        if swap_move(model, routes):
            improved = True

        if or_opt_move(model, routes, max_segment_len=3, max_checks=400):
            improved = True

        if two_opt_star(model, routes, max_checks=400):
            improved = True

        if family_node_replacement(model, routes, max_checks=400):
            improved = True

        if not improved:
            break

    sol = Solution()
    sol.routes = routes
    total_cost = 0
    for r in routes:
        total_cost += r.cost
    sol.cost = total_cost
    return sol



# 5) Local Search mechanisms

def two_opt_route(route, cost_matrix):
# Best-improvement 2-opt inside a single route.
# Reverse a segment to replace edges (a-b, c-d) with (a-c, b-d).

    seq = route.sequence_of_nodes
    best_cost = 0
    best_i = None
    best_k = None

    for i in range(1, len(seq) - 2):
        for k in range(i + 1, len(seq) - 1):
            a = seq[i - 1]
            b = seq[i]
            c = seq[k]
            d = seq[k + 1]

            old_cost = cost_matrix[a.id][b.id] + cost_matrix[c.id][d.id]
            new_cost = cost_matrix[a.id][c.id] + cost_matrix[b.id][d.id]
            dcost = new_cost - old_cost

            if dcost < best_cost:
                best_cost = dcost
                best_i = i
                best_k = k

    if best_cost < 0:
        segment = seq[best_i:best_k + 1]
        segment.reverse()
        seq[best_i:best_k + 1] = segment
        update_route_cost_and_load(route, cost_matrix)
        return True

    return False


def relocate_move(model, routes):
#   Relocate a single customer from one position/route to another.
#   Best-improvement search over all possibilities.

    C = model.cost_matrix
    best_cost = 0
    best_move = None  # (r_from, pos_from, r_to, pos_to)

    # Precompute loads
    loads = []
    for r in routes:
        total = sum(n.demand for n in r.sequence_of_nodes)
        loads.append(total)

    for r_idx, r in enumerate(routes):
        for i in range(1, len(r.sequence_of_nodes) - 1):
            node = r.sequence_of_nodes[i]
            before = r.sequence_of_nodes[i - 1]
            after = r.sequence_of_nodes[i + 1]

            remove_cost = (
                C[before.id][node.id]
                + C[node.id][after.id]
                - C[before.id][after.id]
            )

            for rr_idx, rr in enumerate(routes):
                for j in range(0, len(rr.sequence_of_nodes) - 1):
                    if rr is r and (j == i or j == i - 1):
                        continue

                    prev = rr.sequence_of_nodes[j]
                    nxt = rr.sequence_of_nodes[j + 1]
                    add_cost = (
                        C[prev.id][node.id]
                        + C[node.id][nxt.id]
                        - C[prev.id][nxt.id]
                    )

                    dcost = add_cost - remove_cost
                    if dcost >= best_cost:
                        continue

                    new_load_from = loads[r_idx] - node.demand
                    new_load_to = loads[rr_idx] + node.demand
                    if new_load_from > model.capacity or new_load_to > model.capacity:
                        continue

                    best_cost = dcost
                    best_move = (r_idx, i, rr_idx, j)

    if best_move is None:
        return False

    r_from_idx, pos_from, r_to_idx, pos_to = best_move
    r_from = routes[r_from_idx]
    r_to = routes[r_to_idx]

    node = r_from.sequence_of_nodes.pop(pos_from)
    if r_from_idx == r_to_idx and pos_to >= pos_from:
        pos_to -= 1
    r_to.sequence_of_nodes.insert(pos_to + 1, node)

    for idx in {r_from_idx, r_to_idx}:
        update_route_cost_and_load(routes[idx], C)

    return True


def swap_move(model, routes):
#    Swap two customers between routes (or within the same route).

    C = model.cost_matrix
    best_cost = 0
    best_move = None  # (r1_idx, pos1, r2_idx, pos2)

    loads = []
    for r in routes:
        total = sum(n.demand for n in r.sequence_of_nodes)
        loads.append(total)

    for r1_idx, r1 in enumerate(routes):
        for i in range(1, len(r1.sequence_of_nodes) - 1):
            n1 = r1.sequence_of_nodes[i]
            b1 = r1.sequence_of_nodes[i - 1]
            a1 = r1.sequence_of_nodes[i + 1]

            for r2_idx in range(r1_idx, len(routes)):
                r2 = routes[r2_idx]
                if r2_idx == r1_idx:
                    start_j = i + 1
                else:
                    start_j = 1

                for j in range(start_j, len(r2.sequence_of_nodes) - 1):
                    n2 = r2.sequence_of_nodes[j]
                    b2 = r2.sequence_of_nodes[j - 1]
                    a2 = r2.sequence_of_nodes[j + 1]

                    old_cost = (
                        C[b1.id][n1.id]
                        + C[n1.id][a1.id]
                        + C[b2.id][n2.id]
                        + C[n2.id][a2.id]
                    )
                    new_cost = (
                        C[b1.id][n2.id]
                        + C[n2.id][a1.id]
                        + C[b2.id][n1.id]
                        + C[n1.id][a2.id]
                    )

                    if r1 is r2 and j == i + 1:
                        old_cost = (
                            C[b1.id][n1.id]
                            + C[n1.id][n2.id]
                            + C[n2.id][a2.id]
                        )
                        new_cost = (
                            C[b1.id][n2.id]
                            + C[n2.id][n1.id]
                            + C[n1.id][a2.id]
                        )

                    dcost = new_cost - old_cost
                    if dcost >= best_cost:
                        continue

                    new_load_r1 = loads[r1_idx] - n1.demand + n2.demand
                    new_load_r2 = loads[r2_idx] - n2.demand + n1.demand
                    if new_load_r1 > model.capacity or new_load_r2 > model.capacity:
                        continue

                    best_cost = dcost
                    best_move = (r1_idx, i, r2_idx, j)

    if best_move is None:
        return False

    r1_idx, pos1, r2_idx, pos2 = best_move
    r1 = routes[r1_idx]
    r2 = routes[r2_idx]
    n1 = r1.sequence_of_nodes[pos1]
    n2 = r2.sequence_of_nodes[pos2]

    r1.sequence_of_nodes[pos1] = n2
    r2.sequence_of_nodes[pos2] = n1

    update_route_cost_and_load(r1, C)
    if r2 is not r1:
        update_route_cost_and_load(r2, C)

    return True


def two_opt_star(model, routes, max_checks=400):

#   - cut two routes and exchange their tails.

    C = model.cost_matrix
    depot = model.depot
    checks = 0

    n_routes = len(routes)
    for r1_idx in range(n_routes):
        r1 = routes[r1_idx]
        seq1 = r1.sequence_of_nodes
        if len(seq1) <= 3:
            continue

        for r2_idx in range(r1_idx + 1, n_routes):
            r2 = routes[r2_idx]
            seq2 = r2.sequence_of_nodes
            if len(seq2) <= 3:
                continue

            for i in range(1, len(seq1) - 1):
                for j in range(1, len(seq2) - 1):
                    checks += 1
                    if checks > max_checks:
                        return False

                    prefix1 = seq1[:i + 1]
                    tail1 = seq1[i + 1:-1]
                    prefix2 = seq2[:j + 1]
                    tail2 = seq2[j + 1:-1]

                    new_seq1 = prefix1 + tail2 + [depot]
                    new_seq2 = prefix2 + tail1 + [depot]

                    new_r1 = Route(
                        route_id=r1.id,
                        sequence_of_nodes=new_seq1,
                        capacity=model.capacity,
                        cost=0,
                        load=0,
                    )
                    update_route_cost_and_load(new_r1, C)
                    if new_r1.load > model.capacity:
                        continue

                    new_r2 = Route(
                        route_id=r2.id,
                        sequence_of_nodes=new_seq2,
                        capacity=model.capacity,
                        cost=0,
                        load=0,
                    )
                    update_route_cost_and_load(new_r2, C)
                    if new_r2.load > model.capacity:
                        continue

                    old_cost = r1.cost + r2.cost
                    new_cost = new_r1.cost + new_r2.cost

                    if new_cost < old_cost:
                        routes[r1_idx].sequence_of_nodes = new_seq1
                        routes[r1_idx].cost = new_r1.cost
                        routes[r1_idx].load = new_r1.load

                        routes[r2_idx].sequence_of_nodes = new_seq2
                        routes[r2_idx].cost = new_r2.cost
                        routes[r2_idx].load = new_r2.load

                        return True

    return False


def or_opt_move(model, routes, max_segment_len=3, max_checks=400):

#  Or-opt: move a segment of length 2 to max_segment_len from one position
#  to another (possibly different route).

    C = model.cost_matrix

    loads = []
    for r in routes:
        loads.append(sum(n.demand for n in r.sequence_of_nodes))

    checks = 0

    for r_from_idx, r_from in enumerate(routes):
        seq_from = r_from.sequence_of_nodes
        n_from = len(seq_from)
        num_customers_from = n_from - 2
        if num_customers_from < 2:
            continue

        for seg_len in range(2, max_segment_len + 1):
            if num_customers_from < seg_len:
                continue

            for i in range(1, n_from - seg_len):
                j = i + seg_len - 1
                if j >= n_from - 1:
                    break

                segment = seq_from[i:j + 1]
                seg_demand = sum(node.demand for node in segment)

                before = seq_from[i - 1]
                after = seq_from[j + 1]
                remove_cost = (
                    C[before.id][segment[0].id]
                    + C[segment[-1].id][after.id]
                    - C[before.id][after.id]
                )

                for r_to_idx, r_to in enumerate(routes):
                    seq_to = r_to.sequence_of_nodes
                    n_to = len(seq_to)

                    if r_from_idx != r_to_idx:
                        new_load_to = loads[r_to_idx] + seg_demand
                        new_load_from = loads[r_from_idx] - seg_demand
                        if new_load_to > model.capacity or new_load_from > model.capacity:
                            continue

                    for pos in range(0, n_to - 1):
                        checks += 1
                        if checks > max_checks:
                            return False

                        if r_from_idx == r_to_idx and (pos >= i - 1 and pos <= j):
                            continue

                        prev = seq_to[pos]
                        nxt = seq_to[pos + 1]
                        add_cost = (
                            C[prev.id][segment[0].id]
                            + C[segment[-1].id][nxt.id]
                            - C[prev.id][nxt.id]
                        )

                        dcost = add_cost - remove_cost
                        if dcost < 0:
                            if r_from_idx == r_to_idx:
                                full_seq = routes[r_from_idx].sequence_of_nodes
                                segment = full_seq[i:j + 1]
                                without = full_seq[:i] + full_seq[j + 1:]
                                if pos >= i:
                                    pos -= (j - i + 1)
                                new_seq = without[:pos + 1] + segment + without[pos + 1:]
                                routes[r_from_idx].sequence_of_nodes = new_seq
                                update_route_cost_and_load(routes[r_from_idx], C)
                            else:
                                seq_from_now = routes[r_from_idx].sequence_of_nodes
                                seq_to_now = routes[r_to_idx].sequence_of_nodes
                                segment = seq_from_now[i:j + 1]

                                new_from_seq = seq_from_now[:i] + seq_from_now[j + 1:]
                                new_to_seq = (
                                    seq_to_now[:pos + 1]
                                    + segment
                                    + seq_to_now[pos + 1:]
                                )

                                routes[r_from_idx].sequence_of_nodes = new_from_seq
                                routes[r_to_idx].sequence_of_nodes = new_to_seq

                                update_route_cost_and_load(routes[r_from_idx], C)
                                update_route_cost_and_load(routes[r_to_idx], C)

                            return True

    return False


def family_node_replacement(model, routes, max_checks=400):

# For each family:
#      - find visited nodes of this family
#      - find unvisited nodes of this family
#      - try replacing a visited node with an unvisited node in the same position
#      - check if it reduces cost and respects capacity.

    C = model.cost_matrix

    # map node_id using (route_idx, position)
    node_loc = {}
    for r_idx, r in enumerate(routes):
        seq = r.sequence_of_nodes
        for pos in range(1, len(seq) - 1):
            node = seq[pos]
            node_loc[node.id] = (r_idx, pos)

    selected_ids = set(node_loc.keys())
    checks = 0

    for fam in model.families:
        selected_nodes = []
        unselected_nodes = []

        for node in fam.nodes:
            if node.id in selected_ids:
                selected_nodes.append(node)
            else:
                unselected_nodes.append(node)

        if not selected_nodes or not unselected_nodes:
            continue

        for sel_node in selected_nodes:
            r_idx, pos = node_loc[sel_node.id]
            r = routes[r_idx]
            seq = r.sequence_of_nodes

            load_r = sum(n.demand for n in seq)
            prev = seq[pos - 1]
            nxt = seq[pos + 1]

            for cand in unselected_nodes:
                checks += 1
                if checks > max_checks:
                    return False

                new_load = load_r - sel_node.demand + cand.demand
                if new_load > model.capacity:
                    continue

                old_cost = (
                    C[prev.id][sel_node.id]
                    + C[sel_node.id][nxt.id]
                )
                new_cost = (
                    C[prev.id][cand.id]
                    + C[cand.id][nxt.id]
                )
                dcost = new_cost - old_cost

                if dcost < 0:
                    seq[pos] = cand
                    update_route_cost_and_load(r, C)
                    return True

    return False


# 6. ILS - Called after GRASP
def intensify_with_ils(model, best_sol, rnd, ils_iters=ILS_ITERS):

#    Iterated Local Search:
#      - start from best_sol
#      - repeat 'ils_iters' times:
#          - copy best routes
#          - random_kick
#          - local_search
#          - accept if better

    if best_sol is None:
        return None

    best_routes = clone_routes(best_sol.routes)
    best_cost = best_sol.cost

    for m in range(ils_iters):
        cand_routes = clone_routes(best_routes)
        random_kick(model, cand_routes, rnd, num_moves=KICK_MOVES)
        cand_sol = local_search(model, cand_routes, max_passes=LS_MAX_PASSES)

        if cand_sol.cost < best_cost:
            best_cost = cand_sol.cost
            best_routes = clone_routes(cand_sol.routes)
            if flag:
                print(f"new best cost ils: {best_cost}")

    final = Solution()
    final.routes = best_routes
    final.cost = best_cost
    return final


def random_kick(model, routes, rnd, num_moves=KICK_MOVES):

#   Apply a few random relocate moves to change solution's neighborhood.
#   We ignore cost, only check capacity.

    cost = model.cost_matrix

    for n in range(num_moves):
        if len(routes) == 0:
            break

        r_from_idx = rnd.randrange(len(routes))
        r_to_idx = rnd.randrange(len(routes))

        r_from = routes[r_from_idx]
        r_to = routes[r_to_idx]

        if len(r_from.sequence_of_nodes) <= 2:
            continue

        i = rnd.randrange(1, len(r_from.sequence_of_nodes) - 1)
        node = r_from.sequence_of_nodes[i]

        load_to = sum(n.demand for n in r_to.sequence_of_nodes)
        if load_to + node.demand > model.capacity:
            continue

        pos = rnd.randrange(0, len(r_to.sequence_of_nodes) - 1)

        r_from.sequence_of_nodes.pop(i)
        if r_from_idx == r_to_idx and pos >= i:
            pos -= 1
        r_to.sequence_of_nodes.insert(pos + 1, node)

    for r in routes:
        update_route_cost_and_load(r, cost)


def clone_routes(routes):

# Return a copy (new Route objects, same Node objects) of a list of routes.

    new_routes = []
    for r in routes:
        new_seq = list(r.sequence_of_nodes)
        new_route = Route(
            route_id=r.id,
            sequence_of_nodes=new_seq,
            capacity=r.capacity,
            cost=r.cost,
            load=r.load,
        )
        new_routes.append(new_route)
    return new_routes


# 7. Update_route_cost_and_load
def update_route_cost_and_load(route, cost_matrix):
# Reset and recompute cost and load of a route using the sol_checker logic.
    route.cost = 0
    route.load = 0
    calculate_route_cost_demand(route, cost_matrix)


# 8. save_solution
#Print txt with sample_sol structure
def save_solution(sol, filename):
    with open(filename, "w") as f:
        f.write(f"Cost: {int(sol.cost)}\n")
        for r in sol.routes:
            ids = [str(n.id) for n in r.sequence_of_nodes]
            f.write("-".join(ids) + "\n")
