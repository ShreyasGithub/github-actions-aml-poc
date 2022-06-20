from azureml.core import Workspace, Datastore, Dataset, Run

def evaluate_model():
    run = Run.get_context()
    current_metrics = run.get_metrics()
    print('current_metrics', current_metrics)
    
evaluate_model()