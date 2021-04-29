# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 00:34:24 2021

@author: Nayphilim
"""

from distutils.core import setup
import py2exe
# import aiml
# import os
# import pandas as pd
# import nltk
# import numpy as np
# import re
# import requests
# import json
# import uuid
# import urllib
# import random
# from nltk.stem import wordnet
# from sklearn.metrics import pairwise_distances
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk import pos_tag
# import PySimpleGUI as gui
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.preprocessing import image
# import cv2
# from nltk.sem import Expression
# from nltk.inference import ResolutionProver
# from azure.cognitiveservices.vision.computervision import ComputerVisionClient
# from msrest.authentication import CognitiveServicesCredentials
   
setup(windows=['FishFeederBot.py'],  
      options = {
              "py2exe":{
                  "packages": ["aiml","os", "pandas","nltk","numpy","re","requests","json","uuid","urllib", "random", "cv2", "nltk.stem", "sklearn.metrics","PySimpleGUI","tensorflow", "nltk.sem","nltk.inference", "azure.cognitiveservices.vision.computervision", "msrest.authentication", "py2exe", "distutils.core"]
                  }})