import numpy as np
import json
import pickle
import random
import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

from tkinter import *

nltk.download('punkt')
nltk.download('wordnet')

reducer = nltk.WordNetLemmatizer() #this is to reduce similar words such as study,studying,studied

intents = json.loads(open('korean.json', encoding="utf8").read())


labels = [] #collects all the tags
words = [] #collects every single word in patterns
datas = [] #contains words: tags

def dataCollect(labels,words,datas):  #this function collects data from korean.json for words,data and labels arrays
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            words_pattern = nltk.word_tokenize(pattern) #converts to array or words for each pattern
            words.extend(words_pattern)
            datas.append((words_pattern, intent['tag'])) #words: tag formatt
            if intent['tag'] not in labels:
                labels.append(intent['tag'])
    return labels, words, datas

def saveData(words, labels):    #remove duplicate data sort them and save them using pickels

    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(labels, open('labels.pkl', 'wb'))
trainingData = []

def convertToBinary(numLabels):  #this turns the training straing data to 0,1 so computer can understand
    output_empty = [0] * numLabels
    for data in datas:
        computeData = []
        patterns = data[0]   #words in datas
        patterns = [reducer.lemmatize(word) for word in patterns]
        for word in words:
            computeData.append(1) if word in patterns else computeData.append(0)

        output= list(output_empty)
        output[labels.index(data[1])] = 1

        trainingData.append([computeData, output])


def organizeTrainData(trainingData):    # oranzing training data
    random.shuffle(trainingData)
    training = np.array(trainingData)
    train_x = list(training[:, 0])  # 2d array format every row(horizontal) of first index(:,0)
    train_y = list(training[:, 1])  # 2d array format every row(horizontal) of second index(:,2)

    return train_x,train_y

def train(train_x, train_y):  #training process
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=True)
    model.save('korean.model', hist)
    print( model.evaluate(x_train,y_train))
    print("training finsihed")


everyData = dataCollect(labels,words,datas)
labels = sorted(set(everyData[0]))
words = sorted(set(everyData[1]))
data = everyData[2]
saveData(words,labels)

convertToBinary(len(labels))
x_train , y_train = organizeTrainData(trainingData)

#use this to retrain the model
# train(x_train,y_train)
#/////////////////////////////

#chatbot

#loading the data and the model that was saved before
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))
model = load_model('korean.model')

#to remove similar words
def organize_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [reducer.lemmatize(word) for word in sentence_words]
    return sentence_words

#onverting to binary number 0 and 1 so compute can understand
def bag(sentence):
    sentence_words = organize_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#main function
def click():
    myLabel = Label(root,bg = "yellow", text="You: " + e.get())  #e.get gets the input data
    myLabel.pack()
    bow = bag(e.get()) #convert to 0 1 of input
    respond = model.predict(np.array([bow]))  #using model to predict output text
    result_index = np.argmax(respond)
    tag = classes[result_index]
    #
    for tg in intents['intents']:
         if tg['tag'] == tag:
             responses = tg['responses']

    result = random.choice(responses)
    myLabel = Label(root,bg = "light blue", text="Bot: " + result)
    myLabel.pack()
    e.delete(0, END)

#tkinter designing
root = Tk()
e = Entry(root, width=50)
root.resizable(1200,0)
e.pack()

myButton = Button(root, text="Click Me!", command=click)
myButton.pack()

root.mainloop()

