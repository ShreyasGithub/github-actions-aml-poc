from azureml.core import Workspace, Datastore, Dataset, Run, PipelineRun

def evaluate_model():
    run = Run.get_context()
    pipeline_run = PipelineRun(run.experiment)
    trianing_run = pipeline_run.find_step_run('model training step')
    current_metrics = trianing_run.get_metrics()
    print('current_metrics', current_metrics)
    
evaluate_model()