import os
import azureml.core
from azureml.core import (
    Workspace,
    Experiment,
    Dataset,
    Datastore,
    ComputeTarget,
    Environment,
    ScriptRunConfig
)
from azureml.data import OutputFileDatasetConfig
from azureml.core.compute import AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, StepSequence



def main(workspace):
    
    # config = ScriptRunConfig(source_directory='./code/data_preparation',
    #                          script='data_loader.py',
    #                          compute_target='github-cluster')

    # env = Environment.from_conda_specification(
    #     name='train-env',
    #     file_path='./code/data_preparation/data_prep_env.yml'
    # )
    # config.run_config.environment = env
    
    # return config
    
    data_prep_env = Environment.from_conda_specification(
        name='data_prep_env',
        file_path='./code/data_preparation/data_prep_env.yml'
    )
    
    data_prep_run_config = ScriptRunConfig(
        source_directory='./code/data_preparation',
        compute_target='github-cluster',
        environment = data_prep_env        
    )
    
    data_prep_step = PythonScriptStep(
        name="data preparation step",
        script_name="data_loader.py",
        source_directory=data_prep_run_config.source_directory,
        runconfig=data_prep_run_config.run_config,
    )
    
    train_env = Environment.from_conda_specification(
        name='data_prep_env',
        file_path='./code/train/train_env.yml'
    )
    
    train_run_config = ScriptRunConfig(
        source_directory='./code/train',
        compute_target='github-cluster',
        environment = train_env        
    )
    
    
    train_step = PythonScriptStep(
        name="model training step",
        script_name="train.py",
        source_directory=train_run_config.source_directory,
        runconfig=train_run_config.run_config,
        allow_reuse=False,
    )
    
    evaluation_env = Environment.from_conda_specification(
        name='evaluation_env',
        file_path='./code/evaluation/evaluation_env.yml'
    )
    
    evaluation_run_config = ScriptRunConfig(
        source_directory='./code/evaluation',
        compute_target='github-cluster',
        environment = evaluation_env        
    )
    
    
    evaluation_step = PythonScriptStep(
        name="model evaluation step",
        script_name="evaluate.py",
        source_directory=evaluation_run_config.source_directory,
        runconfig=evaluation_run_config.run_config,
        allow_reuse=False,
    )
    
    
    step_sequence = StepSequence(steps=[
        data_prep_step,
        train_step,
        evaluation_step
    ])
    
    return Pipeline(workspace, steps=step_sequence)

if __name__ == "__main__":
    ws = Workspace.from_config()
    main(ws)