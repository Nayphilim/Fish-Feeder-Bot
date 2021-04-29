# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:27:36 2020

@author: Nayphilim
"""
import aiml
import os
import pandas as pd
import nltk
import numpy as np
import re
import sys
import requests
import json
import uuid
import urllib
import random
from contextlib import contextmanager
from nltk.stem import wordnet
from sklearn.metrics import pairwise_distances
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag
import PySimpleGUI as gui
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import cv2
from nltk.sem import Expression
from nltk.inference import ResolutionProver
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig
read_expr = Expression.fromstring

def fishFeederBot():
           
      #function that normalises sentences and performs lemmatization      
    def text_normalization(text):
        text = str(text).lower() #convert sentence to lower case
        spl_char_text = re.sub(r'[^a-z0-9]', ' ', text) #remove special characters
        txt_tokens = nltk.word_tokenize(spl_char_text) #split sentence into tokens
        lema = wordnet.WordNetLemmatizer() #initialising lemmatizer
        tag_list = pos_tag(txt_tokens, tagset=None)
        lema_words=[] #empty list for lemmatizer words
        for txt_tokens,pos_token in tag_list:
            if pos_token.startswith('V'): #token is a verb
                pos_val ='v'
            elif pos_token.startswith('J'): #token is an adjective
                pos_val ='a'    
            elif pos_token.startswith('R'): #token is an adverb
                pos_val ='r'
            else: #token is a noun
                pos_val ='n' 
            lema_token = lema.lemmatize(txt_tokens,pos_val) #performing lemmatization
            lema_words.append(lema_token) #append lemmatized token into list
        return " ".join(lema_words) #return the lemmatized tokens as a sentence
                    
    def get_card_details(card):
        card = card.replace(" ","-")
        r = requests.get("https://api.scryfall.com/cards/named?exact=" + card)
        if(r.status_code == 200):
            cardDetails = json.loads(r.text)
            return cardDetails
        else:
            return "null"
        
    def convert_colour_format(coloursOld):
        coloursNew = []
        for colour in coloursOld:
            if(colour == 'W'):
                coloursNew.append('white')
            elif(colour == 'U'):
                coloursNew.append('blue')
            elif(colour == 'B'):
                coloursNew.append('black')
            elif(colour == 'R'):
                coloursNew.append('red')
            elif(colour == 'G'):
                coloursNew.append('green')
            
        return coloursNew
    
    #normalizes text for knowledge base
    def normalize_text_kb(text):
        text = text.replace(" ", "_")
        text = text.lower()
        return text
    
    #utilizes azures translation api to detect language entered and translate to english
    def translate_text_en(text,language):
        path = '/translate'
        constructed_url = transEndpoint + path
        
        params = {
            'api-version': '3.0',
            'to': language
        }
        constructed_url = transEndpoint + path
        
        headers = {
            'Ocp-Apim-Subscription-Key': transKey,
            'Ocp-Apim-Subscription-Region': location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

        # You can pass more than one object in body.
        body = [{
            'text': text
        }]
        
        request = requests.post(constructed_url, params=params, headers=headers, json=body)
        if(request.status_code == 200):
            response = json.loads(request.text)
            
            #get language of input
            inputLang = response[0].get('detectedLanguage')
            inputLang = inputLang.get('language')
            #get response in en
            responseText = response[0].get('translations')
            responseText = responseText[0].get('text')
            
        return responseText, inputLang
    
    def read_image_txt(imagePath):
       cardImage = open(imagePath, "rb")
       result = computervision_client.recognize_printed_text_in_stream(cardImage)
       return result
           
    
    def url_to_image(url):
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image,(int(488/2),int(680/2)))
        image = cv2.imencode('.png', image)[1].tobytes()
        return image
    
    def get_artist_artwork(artist):
        if(re.search(r"\s", artist)):
            firstName, lastName = artist.split(" ")
        else:
            firstName = artist
            lastName = ""
        if(lastName != ""):
            query = "https://api.scryfall.com/cards/search?order=name&q=%28artist%3A"+firstName+"+artist%3A"+lastName+"%29"
        else:
            query = "https://api.scryfall.com/cards/search?order=name&q=%28artist%3A"+firstName+"%29"
        
        r = requests.get(query)
        if(r.status_code == 200):
            print("Fish Feeder: Okay heres a random piece by " + artist)
            result = json.loads(r.text)
            cardList = result.get("data")
            rn = random.randint(0,len(cardList)-1)
            card = cardList[rn]
            cardImg = card.get("image_uris")
            cardImg = cardImg.get("art_crop")
            cardImg = urllib.request.urlopen(cardImg)
            cardImg = np.asarray(bytearray(cardImg.read()), dtype="uint8")
            cardImg = cv2.imdecode(cardImg, cv2.IMREAD_COLOR)
            cv2.imshow(artist+" images", cardImg)
        else:
           print("Fish Feeder: Sorry I couldnt find the artist you were looking for") 
           
    
    
    
    cardBack = cv2.imread('card_back.jpg')
    cardBack = cv2.resize(cardBack, (int(488/2),int(680/2)))
    cardBack = cv2.imencode('.png', cardBack)[1].tobytes()
    layout = [[gui.Text("Hey, im the Fish Feeder Bot. Feel free to talk anything MtG related with me!\nI can look at images of magic cards and attempt to determine whether there is a bird, cat, dragon or goblin present in the art")],
                  [gui.Output(size=(127, 50), font=('Helvetica 10')),
                   gui.Image(data=cardBack, key="imageView")],
                  [gui.ML(size=(85, 5), enter_submits=True, key='userInput', do_not_clear=False),
                   gui.Button('SEND', button_color=(gui.YELLOWS[0], gui.BLUES[0]), bind_return_key=True),
                   gui.Input(key='-FILENAME-', visible=False, enable_events=True),
                   gui.FileBrowse('IMAGE', button_color=(gui.YELLOWS[0], gui.GREENS[0]), file_types=(('jpeg files','*.jpg'), ("all files","*.*"))),
                   ]]
    
    window = gui.Window('Fish Feeder Bot', layout, return_keyboard_events=True)
    
    
    #Microsoft Azure Credentials and setup
    transKey = "de5aeee40d644f12890f0a93468345a3"
    transEndpoint = "https://api.cognitive.microsofttranslator.com"
    readerKey = "fecfc6af753649969d53b54518abc7c0"
    readerEndpoint = "https://fish-feeder-bot-card-reader.cognitiveservices.azure.com/"
    speechKey = "1e86c9db7094430c8b8b4e80cfb22b16"
    speechEndpoint = "https://uksouth.api.cognitive.microsoft.com/sts/v1.0/issuetoken"
    location = "westeurope"
    computervision_client = ComputerVisionClient(readerEndpoint, CognitiveServicesCredentials(readerKey))
    speech_config = SpeechConfig(subscription=speechKey, region="uksouth")
    audio_config = AudioOutputConfig(use_default_speaker=True)
    synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    
    # Create the kernel and learn AIML files
    kern = aiml.Kernel()
    kern.setTextEncoding(None)
    
    
    # determine if brain file already exists and load it
    if os.path.isfile("bot_brain.brn"):
        kern.bootstrap(brainFile = "bot_brain.brn")
    else:
        kern.bootstrap(learnFiles="basic_chat.xml")
        kern.saveBrain("bot_brain.brn")
    
    
    
    
    kb=[]
    kbData = pd.read_csv('kb.csv',header=None) 
    [kb.append(read_expr(row)) for row in kbData[0]]
    #give expression that can never be true to test if kb is integral
    expr = read_expr('-Basic_Land (x) -> Land (x)')
    if(ResolutionProver().prove(expr, kb) == True):
        #kb is not intergral
        print('kb is not integral, please fix and try again')
    
    #specify memory device usage for tensorflow
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
  # Invalid device or cannot modify virtual devices once initialized.
      pass
    #init custom cnn model
    model = keras.models.load_model('creature_type_model_bgcd/')
    
    colourList = ["white", "blue", "black", "red", "green"]    
                
    df = pd.read_csv('QASheet.csv')
    df.head()
    df['Lemmatized'] = df['Question'].apply(text_normalization) #clean questions
    df.head()    
    
    
    
    
    
    #answer response loop
    # Press CTRL-C to break this loop
    while True:
        event, value = window.read()
        responseAgent = 'default'
        
        
        
        #event handler for user sending a message
        if event == 'SEND':
            userInput = value['userInput'].rstrip()
            userInputTrans, inputLang = translate_text_en(userInput, 'en')
            print('You: {}'.format(userInput))
            window['userInput'].update('')
            answer = kern.respond(userInputTrans)
            #print('this is the answer: ', answer)
            if answer == "": #if there is no aiml response use cosine similarity
                responseAgent = 'cosSim'
            else:
                responseAgent = 'aiml'
                
        #event handler for image uploaded                
        if event == '-FILENAME-': 
            filename = f'{value["-FILENAME-"]}'
            #if filename is not empty
            if len(filename):
                
                try:
                    img = image.load_img(filename, target_size=(680,488)) 
                    displayImg = cv2.imread(filename)
                    displayImg = cv2.resize(displayImg,(int(488/2),int(680/2)))
                    displayImg = cv2.imencode('.png', displayImg)[1].tobytes()
                except FileNotFoundError as e:
                    break
                print('You: Here is an image of a card, what creature do you think is in the art?')
                window['imageView'].update(data=displayImg)
                
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                
                val = np.argmax(model.predict(x), axis=-1)
                

                if val == 0:
                    print("Fish Feeder: i think your card has a bird in it")
                elif val == 1:
                    print("Fish Feeder: i think your card has a cat in it")
                elif val == 2:
                    print("Fish Feeder: i think your card has a dragon in it")
                elif val == 3:
                    print("Fish Feeder: i think your card has a goblin in it")
                else:
                    print(val)
                    
                
                #read card text
                cardText = read_image_txt(filename)
                if(cardText != "null"):
                    print("Card Text\n-------------------------------------------------------")
                    for region in cardText.regions:
                        for line in region.lines:
                            s = ""
                            for word in line.words:
                                s += word.text + " "
                            print(s)
                    print("-------------------------------------------------------")
                    
        if event is None or event == 'Cancel':
           break 
       
        elif event ==  gui.WIN_CLOSED:            # quit if exit event or X
            window.close()
            break
    
        #pre-process user input and determine response agent (if needed)
                #data frame bow
    
    
    
        
            
        #activate selected response agent
        if responseAgent == 'aiml':
            if answer[0] == '#': #if aiml response is logical component
                params = answer[1:].split('$')
                cmd = int(params[0])
                if cmd == 0:
                    transAns,inputLang = translate_text_en(params[1], inputLang)
                    print(transAns)
                    synthesizer.speak_text_async(transAns)
                    break
                elif cmd == 31: # if input pattern is "I know that * is *"
                    object,subject=params[1].split(' is ')
                    expr1=read_expr(subject + '(' + object + ')')
                    expr2=read_expr(subject + '(-' + object + ')')

                    if(ResolutionProver().prove(expr1, kb) == False and ResolutionProver().prove(expr2, kb) == False):
                        kb.append(expr1)
                        print('Fish Feeder: OK, I will remember that',object,'is', subject)
                        
                    else: 
                       print('Fish Feeder: ', object, 'is not definitely', subject) 
          
                     
                    
                elif cmd == 32: # if the input pattern is "check that * is *"
                    object,subject=params[1].split(' is ')
                    expr=read_expr(subject + '(' + object + ')')
                    if (ResolutionProver().prove(expr, kb) == True):
                        print('Fish Feeder: Correct.')
                    else:
                        #query the NOT of the original query
                        expr=read_expr(subject + '(-' + object + ')')
                        #if -q == true, q is false
                        if(ResolutionProver().prove(expr, kb) == True):
                            print('Fish Feeder: Incorrect')
                        else:
                            print('Fish Feeder: Sorry I do not know')
                elif cmd == 33: # if the input pattern is "check that * is * and *"
                       object,subjects=params[1].split(' is ') 
                       subject1,subject2=subjects.split(' and ')
                       expr=read_expr(subject1 + '(' + object + ')' + ' & ' + subject2 + '(' + object + ')' )
                       if(ResolutionProver().prove(expr, kb) == True):
                           print('Fish Feeder: Correct.')
                       else:
                           expr=read_expr('-' + subject1 + '(' + object + ')' + ' & ' + '-' + subject2 + '(' + object + ')' )
                           if(ResolutionProver().prove(expr, kb) == True):
                               print('Fish Feeder: Incorrect')
                           else:
                               print('Fish Feeder: Sorry I do not know')
                elif cmd == 34: # if the input pattern is "check that * is not *"
                      object,subject=params[1].split(' is not ')
                      expr=read_expr(subject + '(' + object + ')')
                      if (ResolutionProver().prove(expr, kb) == False):
                          print('Fish Feeder: Correct.')
                      else:
                          print('Fish Feeder: Incorrect')
                elif cmd == 35: # if the input pattern is "check that all * are *"
                    object,subject=params[1].split(' are ')
                    expr=read_expr('all x.(' + subject + '(x)' + ' -> ' + object + '(x))')
                    if (ResolutionProver().prove(expr, kb) == True):
                        print('Fish Feeder: Correct.')
                    else:
                        expr=read_expr('all x.(' + subject + '(x)' + ' -> -' + object + '(x))')
                        if (ResolutionProver().prove(expr, kb) == True):
                            print('Fish Feeder: Incorrect')
                        else:
                            print('Fish Feeder: Sorry I do not know')
                elif cmd == 36: # if the input pattern is "check that  * has type *"
                    object,subject=params[1].split(' type ')
                    expr=read_expr('type(' + object + ','+ subject + ')')
                    if (ResolutionProver().prove(expr, kb) == True):
                        print('Fish Feeder: Correct.')
                    else:
                        print('Fish Feeder: Incorrect')
                elif cmd == 37: # if the input pattern is "check that  * is colour *"
                    object,subject=params[1].split(' colour ')
                    card = object.replace(" ", "_")
                    card = card.lower()
                    subject = subject.lower()
                    expr = read_expr(subject + '(' + card + ')')
                    
                    if (ResolutionProver().prove(expr, kb) == True):
                        print('Fish Feeder: Correct.')
                    else:
                        cardDetails = get_card_details(object)
                        if(cardDetails != 'null'):
                            cardColourList = cardDetails.get('colors')
                            cardColours = convert_colour_format(cardColourList)
                            for colour in colourList:
                                if(colour in cardColours):
                                    kb.append(read_expr(colour + '(' + card + ')'))
                                else:
                                    kb.append(read_expr(colour + '(-' + card + ')'))

                            if (ResolutionProver().prove(expr, kb) == True):
                                print('Fish Feeder: Correct.')
                            else:
                                 #query the NOT of the original query
                                expr = read_expr(subject + '(-' + card + ')')
                                
                                #if -q == true, q is false
                                if(ResolutionProver().prove(expr, kb) == True):
                                    print('Fish Feeder: Incorrect')
                                else:
                                    print('Fish Feeder: Sorry I do not know')
                        else:
                            print('Fish Feeder: Sorry i didnt quite get what card you were talking about, please try again')
                
                            
                elif cmd == 38: # if the input pattern is asking what * is
                    card = params[1]
                    cardDetails = get_card_details(card)
                    cardImgUrlList = cardDetails.get('image_uris')
                    cardImgUrl = cardImgUrlList.get('normal')
                    
                    #fetch card image and display
                    cardImg = url_to_image(cardImgUrl)
                    window['imageView'].update(data=cardImg)
                    
                    cardName = str(cardDetails.get('name'))
                    cardColour = cardDetails.get('color_identity')
                    cardColour = convert_colour_format(cardColour)
                    cardColour = ', '.join(cardColour)
                    cardType = str(cardDetails.get('type_line'))
                    cardText = str(cardDetails.get('oracle_text'))
                    
                    
                    cardBreakdown = cardName + " is a "+cardColour+" "+cardType+" with the following oracle text:\n\n "+cardText+"\n\n I have displayed the image of the card on the right"
                    
                    print('Fish Feeder:', cardBreakdown)
                elif cmd == 39: # if the input pattern is asking what * is worth
                    card = params[1]
                    cardDetails = get_card_details(card)
                     #fetch card image and display
                    cardImgUrlList = cardDetails.get('image_uris')
                    cardImgUrl = cardImgUrlList.get('normal')
                    cardImg = url_to_image(cardImgUrl)
                    window['imageView'].update(data=cardImg)
                    
                    cardName = str(cardDetails.get('name'))
                    cardPrices = str(cardDetails.get('prices'))
                    
                    print('Fish Feeder:', cardName + " is worth the following:\n\n " + cardPrices)
                    
                elif cmd == 40: #if the input pattern is asking for artwork from *
                    artist = params[1]
                    get_artist_artwork(artist)
                elif cmd == 41: # if the input pattern is asking for artwork from the artist of *
                    card = params[1]
                    cardDetails = get_card_details(card)
                    cardArtist = str(cardDetails.get('artist'))
                    get_artist_artwork(cardArtist)
                    
                       
            else:   
                transAns, inputLang = translate_text_en(answer, inputLang)
                print('Fish Feeder: ', transAns)
                synthesizer.speak_text_async(transAns)
            
        if responseAgent == 'cosSim':
           
            tfidf = TfidfVectorizer() #init term frequency vectorizer
            x = tfidf.fit_transform(df['Lemmatized']).toarray()
            #return all unique words from data
            features = tfidf.get_feature_names()
            df_tdidf = pd.DataFrame(x,columns = tfidf.get_feature_names())
            df_tdidf.head()
            #user input bow
            # stop = stopwords.words('english') #collection of all stopwords
            # Q = []
            # a = userInputTrans.split()
            #loop that removes all stopwords
            #currently removed as it does not work
            # for i in a:
            #     if i in stop: #if current word is a stopword, skip to next
            #         continue
            # else:
            #     Q.append(i)
            #     b = " ".join(Q) #b = sentence without stopwords
            Question_lemma = text_normalization(userInputTrans) #lemmatized version of question
            Question_tdidf = tfidf.transform([Question_lemma]).toarray() #question tfidf
            
            #cosine similarity calculation
            cos_val = 1 - pairwise_distances(df_tdidf, Question_tdidf, metric = 'cosine')
            index_value = cos_val.argmax()
            answer = df['Answer'].loc[index_value]
            transAns,inputLang = translate_text_en(answer, inputLang)
            print('Fisher Feeder: ',transAns)
            synthesizer.speak_text_async(transAns)
    
fishFeederBot()       
        



        

