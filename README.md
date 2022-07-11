# Schau mir in die Augen

People recognition based on eye movements.

Take a look in [OVERVIEW](documentation/OVERVIEW.md) to get a feeling for this project or follow the steps in [SETUP](documentation/SETUP.md) and [EVALUATION](documentation/EVALUATION.md) to start right away.

[Overview of all Documentation Files](documentation/menu.md)

This Readme is in competition with the Markdown Documentation and the API Documentation. None of it is complete. This should be improved. If your are new, you may prefer the documentation files first (especially if you don't use Docker).

# Experiments documentation for Gender prediction

Gender prediction based on eye movements:
To run the experiments of the main three groups of the Gender prediction paper:

- `make Mixed_Group_accuracy`

- `make Non_dyslexic_Group_accuracy`

- `make Dyslexic_Group_accuracy`

To return the average accuracy over the runs (seeds) of the main three groups of the Gender prediction paper:

- `make Mixed_Group_average_accuracy`

- `make Non_dyslexic_Group_average_accuracy`

- `make Dyslexic_Group_average_accuracy`

Take a look in [Gender_Prediction_Experiments](documentation/gender_classification_accuracy)  to get the information on our results and how to get the accuracies for Gender classification in different experiments.

# Experiments documentation for User Identification

People recognition prediction based on eye movements.

Take a look in [User_Identification_Experiments](documentation/user_identification_accuracy) to get the information on our results and how to get these final accuracies for User identification paper.




## Setup development environment

We setup the development environment in a Docker container with the following command.

- `make init`

This command gets the resources for training and testing, and then prepares the Docker image for the experiments.
After creating the Docker image, you run the following command.

- `make create-container`

The above command creates a Docker container from the Docker image which we create with `make init`, and then
login to the Docker container. Now we made the development environment. For create and evaluate the model,
you run the following command.

## Development with Docker container

This section shows how we develop with the created Docker container.

### Edit source code

Most of the source codes of this project, `Schau mir in die Augen` are stored in the `schau_mir_in_die_augen` directory.
Generated Docker container mounts the project directory to ``/work`` of the container and therefore
when you can edit the files in the host environment with your favorite editor
such as Vim, Emacs, Atom or PyCharm. The changes in host environment are reflected in the Docker container environment.

### Update dependencies

When we need to add libraries in `Dockerfile` or `requirements.txt`
which are added to working environment in the Docker container, we need to drop the current Docker container and
image, and then create them again with the latest setting. To remove the Docker the container and image, run `make clean-docker`
and then `make init-docker` command to create the Docker container with the latest setting.

### Login Docker container

Only the first time you need to create a Docker container, from the image created in `make init` command.
`make create-container` creates and launch the schau_mir_in_die_augen container.
After creating the container, you just need run `make start-container`.

### Logout from Docker container

When you logout from shell in Docker container, please run `exit` in the console.

### Run linter

When you check the code quality, please run `make lint`

### Run test

When you run test in `tests` directory, please run `make test`

### Sync data source to local data directory

When you want to download data in remote data sources such as S3 or NFS, `sync-from-remote` target downloads them.

### Sync local data to remote source

When you modify the data in local environment, `sync-to-remote` target uploads the local files stored in `data` to specified data sources such as S3 or NFS directories.

### Show profile of Docker container

When you see the status of Docker container, please run `make profile` in host machine.

### Use Jupyter Notebook

To launch Jupyter Notebook, please run `make jupyter` in the Docker container. After launch the Jupyter Notebook, you can
access the Jupyter Notebook service in http://localhost:8888.

# Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [cookiecutter-docker-science](https://docker-science.github.io/) project template.
