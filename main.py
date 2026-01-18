import time
from Solver import build_solution, save_solution, SELECTED_SEED

if __name__ == "__main__":
    start_time = time.time()
    model,sol = build_solution(seed=SELECTED_SEED)
    end_time = time.time()
    run_time = end_time - start_time

    print(f"Final cost: {sol.cost}")
    print(f"Runtime: {run_time:.2f} seconds")

    save_solution(sol, "Final_Solution.txt")