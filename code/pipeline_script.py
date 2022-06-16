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
from azureml.pipeline.core import Pipeline



def main(workspace):
    
    # config = ScriptRunConfig(source_directory='./code/data_preparation',
    #                          script='data_loader.py',
    #                          compute_target='github-cluster')

    # # set up pytorch environment
    # env = Environment.from_conda_specification(
    #     name='train-env',
    #     file_path='./code/data_preparation/data_prep_env.yml'
    # )
    # config.run_config.environment = env
    
    # return config
    # set up pytorch environment
    env = Environment.from_conda_specification(
        name='train-env',
        file_path='./code/data_preparation/data_prep_env.yml'
    )
    
    run_config = ScriptRunConfig(
        source_directory='./code/data_preparation',
        compute_target='github-cluster',
        environment = env        
    )
    
    data_prep_step = PythonScriptStep(
        name="data preparation step",
        script_name="data_loader.py",
        source_directory=run_config.source_directory,
        runconfig=run_config.run_config,
    )
    
    return Pipeline(workspace, steps=[data_prep_step])

if __name__ == "__main__":
    ws = Workspace.from_config()
    main(ws)