#final code
import text2emotion as te
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
import json
import random
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

class emotionBot:
    
    def __init__(self,text,channel,user):
        self.channel=channel
        self.text=text
        self.user=user


    # @staticmethod
    # def clean_up_sentence(sentence):
    #     # tokenize the pattern - split words into array
    #     sentence_words = nltk.word_tokenize(sentence)
    #     # stem each word - create short form for word
    #     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    #     return sentence_words

    # # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

    # @staticmethod
    # def bow(sentence, words, show_details=True):
    #     # tokenize the pattern
    #     sentence_words = clean_up_sentence(sentence)
    #     # bag of words - matrix of N words, vocabulary matrix
    #     bag = [0]*len(words)  
    #     for s in sentence_words:
    #         for i,w in enumerate(words):
    #             if w == s: 
    #                 # assign 1 if current word is in the vocabulary position
    #                 bag[i] = 1
    #                 if show_details:
    #                     print ("found in bag: %s" % w)
    #     return(np.array(bag))

    # @staticmethod
    # def predict_class(sentence, model):
    #     # filter out predictions below a threshold
    #     p = bow(sentence, words,show_details=False)
    #     res = model.predict(np.array([p]))[0]
    #     ERROR_THRESHOLD = 0.25
    #     results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    #     # sort by strength of probability
    #     results.sort(key=lambda x: x[1], reverse=True)
    #     return_list = []
    #     for r in results:
    #         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    #     return return_list

    # @staticmethod
    # def getResponse(ints, intents_json):
    #     tag = ints[0]['intent']
    #     list_of_intents = intents_json['intents']
    #     for i in list_of_intents:
    #         if(i['tag']== tag):
    #             result = random.choice(i['responses'])
    #             break
    #     return result

    # @staticmethod
    # def chatbot_response(msg):
    #     ints = self.predict_class(msg, model)
    #     res = getResponse(ints, intents)
    #     return res

    
    def get_emotion(self):

        emotion = te.get_emotion(self.text)
        list_emotions = list(emotion.values())

        list_of_emotion_word=[]
        for key in emotion:
            list_of_emotion_word.append(key)

        res_emotion_word=[]
        res_emotion=[]

        res = {}

        for i in range(5):
            if list_emotions[i]!=0:
                res[list_of_emotion_word[i]]=list_emotions[i]
    
        # result_values = list(res.values())
        # for i in range(1):
        #     response = self.chatbot_response("happy")

                
        text = f"The emotion of the text is {res} The possible responses can be: Really happy for you!,We are glad to be of help,How could I help you?, What I can do to make it better?"

        return {"type": "section", "text": {"type": "mrkdwn", "text": text}},

    def get_message_payload(self):
        return{
            "channel":self.channel,
            "blocks":[
                *self.get_emotion(),
            ],
            "user":self.user
        }

    