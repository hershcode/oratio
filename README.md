# Oratio Speech Classifier

This project was made to explore the applications of deep learning in speech recognition. 
I sourced the speech commands dataset made available by Google's Tensorflow project. 
In this project, I did an EDA of speech commands to understand the nature of speech commands. This included analyzing the frequency, signal, and Mel-frequency cepstral coefficients among other data analysis techniques. I applied a simple Neural Network to build a multi-class classification model that could take a ".wav" file as an input and the predict what the appropriate label is. Additionally, I experimented with Vertex AI's automl capabilities. Later on I deployed this model in a Flask app with giving optionality of storing the model locally or using a real-time Vertex AI endpoint. 

# Requirements
- [Python 3.9.10](https://www.python.org/downloads/release/python-3910/)
- [librosa](https://librosa.org/) 
- [Tensorflow](https://www.tensorflow.org/)  
- [malaya-speech](https://github.com/huseinzol05/malaya-speech) 
- [flask](https://flask.palletsprojects.com/en/2.2.x/)

# How to Use it

## Environment
Create a virtual environment:
```
python -m venv oratio
```
 Activate environment using:
```
source oratio/bin/activate
```

Run the following to install requirements:
```
pip install -r requirements.txt
```
## Dataset
Download speech commands to data/raw/

Run the following to generate the datasets:
```
python file.py
```

This will generate 3 csv files: training.csv, validation.csv, test.csv

## Train
Go to notebooks/ and open modeling.ipynb. Run cells to build classification model.
You could also opt to take datasets from previous step into Vertex AI and use AutoML.

## Deploy
You can put model locally in model/ or you can host a real-time endpoint in Vertex AI. 
Specify model path in config.yaml. 

Run flask app:
```
python src/app.py
```