# python-template

## Setup

### Linux and Mac Users

- run the setup script `./setup.sh` or `sh setup.sh`

### Windows Users

- run the setup script `.\setup.ps1`

## Development

- activate python environment: `source .venv/bin/activate`
- run python script: `python <filename.py> `, e.g. `python train.py`
- install new dependency: `pip install sklearn`
- save current installed dependencies back to requirements.txt: `pip freeze > requirements.txt`

## Docker

- build the docker container using `docker-compose build` (You need to make sure that docker has enough memory to build the image)
- start the jupyter lab docker container using `docker-compose up`
- Copy the link (incl. token) from the console and paste it into the browser

## Git LFS

- In order to use git lfs, please refer to the [official instructions](https://git-lfs.github.com/)
- Configure which files to store in Git LFS using the `git lfs track "*.file_ending"`command
