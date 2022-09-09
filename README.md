# docker-mlflow-minio

**This repository is corresponding to [this detailed documentation](https://ikespand.github.io/posts/mlflow/).**

As the name suggest, a `docker-compose` setup to run [MLFlow](https://mlflow.org/) server with MinIO for the artifacts and SQLite for the database for runs.

## How to use it?
0. Clone the repository `git clone https://github.com/ikespand/docker-mlflow-minio.git`.
1. [Optionally] You can configure `.env` file to override default settings or to see current configuration for tracking server and MinIO. Don't be confuse with AWS as this is  coming from [MinIO](https://github.com/minio/minio) which offers S3-like storage facility locally. 
2. Start the docker containers for MinIO and MLFlow with SQLite by `docker-compose up` (`-d` for the background). Use Powershell on Windows OS as volume mounting can have problems with `git-bash`.
3. As a result, you should able to see [localhost:5000](http://localhost:5000/) for our MLFlow server while [localhost:9001](http://localhost:9001/) for MinIO. Also, you will notice that `mlflow_data` and `minio_data` folders have created in this folder. You can also configure this for your next run by editing `docker-compose.yml`

## Log a sample ML experiment
- Once things are running and you're able to see the dashboards for both MinIO and MLFlow then you can proceed to test this. 
- Configure the environment variable so that MLFlow's python library can pick up these to communicate with the MinIO. Open the `bash_profile` and copy-paste the credential from the `.env` file there. We will need to paste `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `MLFLOW_S3_ENDPOINT_URL`.
- Then, you can test the attached script `test_setup_with_scikitlearn.py` for a sample scikit-learn program. If you haven't change any settings in our setup then it should work as long as you have required packages. You can find more details about various things on [abovementioned blog](https://ikespand.github.io/posts/mlflow/).

