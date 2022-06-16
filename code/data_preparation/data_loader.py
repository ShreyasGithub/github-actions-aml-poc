from sklearn import datasets

def load_data():
    # Load iris dataset
    iris_data_frame = datasets.load_iris(as_frame=True)
    print(iris_data_frame.info())
    print(iris_data_frame.head())
    
load_data()
