# Full-Stack Machine Learning

This repository contains the supplementary material for the Full Stack Machine Learning Course (e.g. Digethic Data Scientist / AI-Engineer).

All notebooks under /notepads are structured and can be identified via the folder number and notebook code. All notebooks correspond to the slides and videos produces for this course.

![image](https://user-images.githubusercontent.com/29402504/137859990-054ce9a4-f2d2-4054-8d25-faae4a466c5f.png)

E.g. this identifier referes to folder 2 and notebook with code EDA.



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
