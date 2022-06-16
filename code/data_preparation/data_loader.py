from sklearn import datasets
from azureml.core import Workspace, Datastore, Dataset

def load_data():
    # Load iris dataset
    iris_data_frame = datasets.load_iris(as_frame=True).frame
    print(iris_data_frame.info())
    print(iris_data_frame.head())
    # iris_data_frame.to_csv('github_iris_dataset.csv', index=False)
    workspace = Workspace.from_config()
    Dataset.Tabular.register_pandas_dataframe(iris_data_frame,
        workspace.get_default_datastore(),
        "github_iris_dataset", show_progress=True)
    
load_data()
