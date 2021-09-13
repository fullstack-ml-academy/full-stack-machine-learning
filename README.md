# python-template
## Setup

### Linux Users

- create new python environment: `python3 -m venv .venv`
- activate python environment: `. .venv/bin/activate`
- update the pip version: `pip install --upgrade pip`
- install dependencies: `pip install -r requirements.txt`

### Windows Users

- create new python environment: `python -m venv .venv`
- activate python environment: `.\.venv\Scripts\Activate.ps1`
- update the pip version: `pip install --upgrade pip`
- install dependencies: `pip install -r requirements.txt`

## Development

- activate python environment: `source .venv/bin/activate`
- run python script: `python <filename.py> `, e.g. `python train.py`
- install new dependency: `pip install sklearn`
- save current installed dependencies back to requirements.txt: `pip freeze > requirements.txt`

## Docker

- build the docker container using `docker-compose build` (You need to make sure that docker has enough memory to build the image)
- start the jupyter lab docker container using `docker-compose up`
- Copy the link (incl. token) from the console and paste it into the browser
