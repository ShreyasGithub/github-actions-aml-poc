# run-pytorch.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig


def main(workspace):
    
    config = ScriptRunConfig(source_directory='./code/train',
                             script='train.py',
                             compute_target='github-cluster')

    # set up pytorch environment
    env = Environment.from_conda_specification(
        name='pytorch-env',
        file_path='pytorch-env.yml'
    )
    config.run_config.environment = env

    # run = experiment.submit(config)

    # aml_url = run.get_portal_url()
    # print(aml_url)
    
    return config

if __name__ == "__main__":
    ws = Workspace.from_config()
    main(ws)