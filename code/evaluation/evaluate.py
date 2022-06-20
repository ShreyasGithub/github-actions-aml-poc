from azureml.core import Workspace, Datastore, Dataset, Run
from azureml.core.model import Model as AMLModel
from azureml.pipeline.core import PipelineRun


def evaluate_model():
    run = Run.get_context()
    pipeline_run = PipelineRun(run.experiment, run.parent.id)
    print(list(pipeline_run.get_steps()))
    trianing_run = pipeline_run.find_step_run('model training step')
    current_metrics = trianing_run[0].get_metrics()
    print('current_metrics', current_metrics)
    current_accuracy = current_metrics['Accuracy']
    
    workspace = run.experiment.workspace
    models = AMLModel.list(workspace, name='github-iris-clf', latest=True)
    print(models)
    
evaluate_model()