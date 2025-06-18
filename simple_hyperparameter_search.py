# simple_hyperparameter_search.py
import itertools
import os

# This is a conceptual script. The actual training logic will be in train_final_model.py
# You would call a main training function from here.

def run_training_session(lr, weight_decay, dropout):
    """
    A placeholder function that would launch a training run.
    In a real scenario, this would call the main function of `train_final_model.py`
    with these hyperparameters as arguments.
    """
    print("-" * 50)
    print(f"STARTING TRAINING RUN")
    print(f"  Learning Rate: {lr}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  Dropout (in model): {dropout}") # Note: This requires model modification
    
    # Example of how you might run it
    # command = f"python train_final_model.py --lr {lr} --weight_decay {weight_decay}"
    # os.system(command)
    
    print("Pretending to run training... Done.")
    print("-" * 50)


def perform_grid_search():
    learning_rates = [1e-4, 5e-5]
    weight_decays = [1e-2, 1e-3]
    dropouts = [0.3, 0.5] # This is just conceptual

    # Create all combinations of hyperparameters
    param_grid = list(itertools.product(learning_rates, weight_decays, dropouts))
    
    print(f"Starting Grid Search with {len(param_grid)} combinations.")

    for params in param_grid:
        lr, wd, dr = params
        run_training_session(lr, wd, dr)

if __name__ == '__main__':
    perform_grid_search()