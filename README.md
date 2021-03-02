# Lung Disease Detection from Chest X-ray

## Table of contents
* [Introduction](#introduction)
* [Technologies](#technologies)
* [Setup](#setup)
* [Usage](#usage)

## Introduction
The goal of this project to build a web and mobile app which will diagnose lung diseases from chest x-ray image using deep learning. It will also guide radiologists to make fast and accurate diagnose by segmented and showing heat-map of the chest X-Ray images prior to classifying them.

	
## Technologies
Project is created with:
* Python: 3.8
* Tensorflow: 2.4.1
* Flask: 1.1.2
* React: 17.0.1
* Flutter: 1.22.6
	
## Setup
To run this project, install it locally using pip:

```
$ mkvirtualenv attendance
$ workon attendance
$ pip install requirnments.txt

$ cd ./web_app
$ npm install
```

## Usage
### Train model 
```
$ cd ./chest_xray
$ python build_dataset.py
$ python train.py
$ python test.py
$ python predict.py --image example.png --model model.hdf5

$ run the Lung Disease Detection jupyter notebook (alternative)

```

### Test the web app
```
$ cd ./web_app
$ npm start

On new terminal
$ export FLASK_APP=predict_app.py
$ python -m flask run
```

