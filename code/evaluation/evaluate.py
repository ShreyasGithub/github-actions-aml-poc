from azureml.core import Workspace, Datastore, Dataset, Run, get_run
from azureml.core.model import Model
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
    models = Model.list(workspace, name='github-iris-clf', latest=True)
    
    if len(models):
        print('model_run_id', models[0].run_id)
        previous_run = get_run(run.experiment, models[0].run_id)
        print('previous run metrics', previous_run.get_metrics())
        previous_metrics = previous_run.get_metrics()
        previous_accuracy = previous_metrics['Accuracy']
        
        if current_accuracy < previous_accuracy:
            raise Exception(
                f"current metrics failed to beat champion, current acc:{current_accuracy}, previous acc:{previous_accuracy}"
            )
        else:
            print(f"current metrics beats champion, current acc:{current_accuracy}, previous acc:{previous_accuracy}")
    
    
    
evaluate_model()