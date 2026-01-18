from sol_checker import (
    Route,
    Solution,
    create_model,
    calculate_route_cost_demand,
)

instance = "instance.txt"


def load_model(instance_file: str = instance):
    return create_model(instance_file)
