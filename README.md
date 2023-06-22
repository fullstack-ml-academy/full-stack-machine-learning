# Full-Stack Machine Learning

## Intro

This repository contains the supplementary material for the Full Stack Machine Learning Course (e.g. Digethic Data Scientist / AI-Engineer).

All notebooks under `/notepads` are structured and can be identified via the folder number and notebook code. All notebooks correspond to the slides and videos produces for this course.

![image](https://user-images.githubusercontent.com/29402504/137859990-054ce9a4-f2d2-4054-8d25-faae4a466c5f.png)

E.g. this identifier referes to folder 2 and notebook with code EDA.

## Setup

### Linux and Mac Users

- run the setup script `./setup.sh` or `sh setup.sh`

### Windows Users

- run the setup script `.\setup.ps1`
- if running the script does not work due to access rights, try following command in your terminal: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Development

- Mac/Linux: activate python environment: `source .venv/bin/activate`
- Windows: activate python environment: `.\.venv\Scripts\Activate.ps1`
- run python script: `python <filename.py> `, e.g. `python train.py`
- install new dependency: `pip install sklearn`
- save current installed dependencies back to requirements.txt: `pip freeze > requirements.txt`
- to start Jupyter lab run `jupyter lab --ip=127.0.0.1 --port=8888`


# python is cool...ppp


