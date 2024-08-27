import mlflow


def calculate_sum(x,y):
    return x+y




if __name__ == '__main__':
    # Starting the ML-Flow Sserver :
    with mlflow.start_run():
        x,y=10,20
        # Experiment Tracking
        z=calculate_sum(x,y)
        mlflow.log_param("x",x)
        mlflow.log_param("y",y)
        mlflow.log_metric("z",z)