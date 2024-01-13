import random  # Brings in a toolkit for creating unpredictable results, like flipping coins or rolling a dice  
import numpy as np  # brings a powerful library for working with numbers, espesically large sets of data
import tensorflow as tf # imports TensorFlow a powerful library for creating and training artificial intelligence models. 
from tensorflow.keras.models import Sequential #helps in building model in step-by-step fashion like stacking blocks to create a structure
from tensorflow.keras.layers import LSTM, Dense, Activation # imports three types of layers LTSM(it helps the model remember important patterns in sequences of data, like words in a sentence), Dense (it allows neurons to communicate and process information in a complex way), Activation (it controls how neurons activate and pass information, shaping the models decision making)
from tensorflow.keras.optimizers import RMSprop # imports a technique called RMSprop which helps the model learn effictevily and efficiently. 

filepath = tf.keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt') # grabs a copy of a text file containing shakespeare's work from a specific location and stores path to a file in a variable called [filepath] 

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower() # opening the file, reading the file, decoding it from a specific encoded version and converting all of its text to lower case

text = text[300000:800000] # slices portion of the text file strating from 300,000 to 800,000 

characters = sorted(set(text)) # extracting all individual characters from the selected text from the file organizing it alphabetically and stores them in a variable characters 

char_to_index = dict((c, i) for i, c in enumerate(characters)) # creates a dictionary where each character is assinged a unique numerical code. (its like translating letters to numerical values for model to understand )
index_to_char = dict((i, c) for i, c in enumerate(characters)) # creates a dictonary where char_to_index dict is reversed allowing you to translate numbers back into characters. (its like having a key to read the models output)

seq_length = 40 # the model will process text in chunks of 40 characters at a time
step_size = 3 # the model will slide over three text at a time when creating a training example

sentences = [] # a empty list where it will store segments of texts to be used for training the model. its like setting up a workspaces fpr the models learning
next_characters = [] # a empty list which will store characters that comes immediately after each segment of the text. its like creating a refrence for what the model should predict

for i in range(0, len(text) - seq_length, step_size): # starts loops that systematically moves through text, starting from the beginning and ending just before the last 40 characters. it jumps three character at a time like a focused reader scanning through the text.(taking a stroll through the book, flipping pages every few pages)  
    sentences.append(text[i: i + seq_length]) # slices out chunk of text 40 char long, starting from the current postion (i) and adds it to (sentences) list. (its like creating a flashcard with a short phrase for a model to study) 
    next_characters.append(text[i + seq_length]) # takes the character of text which comes after the sliced chunk of text (the one position i+seq_length) and adds it to the (next_character) list. its like creating a answers for a flashcards so the model can learn what comes next.

x = np.zeros((len(sentences), seq_length, len(characters)), dtype=np.bool) # creates a 3d structure which acts like a collection of character grids with each grid able to hold characters(seq_length) and represnt all the possible characters in text(len(characters)). its like making a bunch of blank character forms to model to fill in.
y = np.zeros((len(sentences), len(characters)), dtype=np.bool) # creates 2d structure which has a row for each sentences and a column for each possible character. its like preparing answer sheets to record which character the model predicts should come next

for i, sentence in enumerate(sentences): #starts a loop that goes through each sentence in the (sentence) list, keeping track of both sentences index (i) and the sentence itself(sentence). its like opening each character grid from one by one. 
    for t, character in enumerate(sentence): #starts a second loop that goes through each character within current sentence, keepin track of both the character's position within the sentence (t) and the actual character itself (character). its like going through each character block on the form.
        x[i, t, char_to_index[character]] = 1 #  fills in the  (x) structre with a '1' like checking a box at the specific position that corresponds to current character. it uses the (char_to_index) dict to translate the character into the numerical code. its like ticking correct character checkbox on the form.
    y[i, char_to_index[next_characters[i]]] = 1 # fills in the (y) structure with '1' at the postion that corresponds to the next character that should follow the sentence. its like circling the predicted next character on the answer sheet. 

model = Sequential() # creates a empty model using Sequential method which allows you to stack layers of the model one by one. (building a blank model assembly line)
model.add(LSTM(128, input_shape=(seq_length, len(characters)))) # adding a memory exoert to model LTSM which is capable of learning long-term dependencies. the 128 part is the memory unit part. and the input shape tells it what kind of input to expect(40 char ata  time, all possible char). its like adding a memory part to assembly line 
model.add(Dense(len(characters))) # its like giving a decision maker to assembly line the (len(characters)) part tells it to make a decisions for each possible character, predicting which one should come next
model.add(Activation('softmax')) # adds a softmax activation layer which converts models prediction into probabilities ensuring that all predicted probabilities add upto 1 its like adding a final vote cating machine to assembly line. 
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01)) # configures the models learning process by defining: (loss:- which measures how the model is  doing during training), (optimizer:- controls how the model learns and adjusts its weight), (learning rate:- it determines how the model adjusts its weights during the training). its like setting the training  goals and choosing a coach for the model. 
model.fit(x, y, batch_size=256, epochs=4) #starts actual training proccess using the character grid(x), and answer sheets(y) to teach the model to predict next character in sequence. it goes through data multiple times (epochs=4) and it processes data in smaller chunks(batch_size=256). its like sending model to training school for several semesters.
model.save('textgenerator.model') #saves the trainend model to a file called "textgenerator.model"

def sample(preds, temperature=1.0): #defines a funtion(sample), consists of two things (preds:- a list of predicted probabilitties for each character), (temperature:- how much randomness is added to a choice)
    preds = np.asarray(preds).astype('float64') # converts (preds) into np array with a specific datatype (float64). its like tto uniformransforming the voting ballot into uniform format for easier calculations.
    preds = np.log(preds) / temperature # applies a transformation to the predicted probabilties, using the temprature value to control how much the scores spreads out. higher (temprature) values make the score more even, leading to more unpredictible choices. lower (temprature) makes the scores more distinct, leading to more predictable choices. 
    exp_preds = np.exp(preds) # applies the exponential function to the transformed scores. its like rescaling the adjusted scores to make them easier to compare.
    preds = exp_preds / np.sum(exp_preds) # normalizes the score by dividing them by their total total sum, ensuring that they represent true probabilities that add upto 1. (converting the scores into true probabilities that add upto 100%)
    probas = np.random.multinomial(1, preds, 1) # draws a random multinomial distribution based on the provided probablities (preds) essentially picking a character at random but with te likelihood of each character being proportional to its adjusted scores. its like drawing a character from a hat based on the adjusted probabilities. 
    return np.argmax(probas) # returns the index of the character with highest probability in the drawn sample, effectively seleccting the most likely character as the final choice. (announcing the winner of the character drawing)

def generate_text(length, temperature): # defines a function (generate_text) which consists of two things (length:- desired length of the generated text), (temperature:- controls how predictable the text will be). its like creating reciepe for writing a story with specifc length and level of creativty
    start_index = random.randint(0, len(text) - seq_length - 1) # chooses a random starting point within the text ensuring that generated text wont start from the same place.(flipping through the book to find a random starting point) 
    generated = '' # creates empty string where genrated text will be stored
    sentence = text[start_index: start_index + seq_length] # extracts a chunk of 40 character long starting from the randomly chosen starting point serving as the initial seed for the generation process. (copying a snippet from a book to use as the first part of the story)
    generated += sentence # appends the extracted sentence to the generated string. (writing the first few sentence onto the blank page)
    for i in range(length): #starts a loop that repeats for the specificed length. its like writing a story with a specific number of character and taking it one step at a time.
        x = np.zeros((1, seq_length, len(characters))) # creates a new 3d structure called (x) using numpy, with dimensionally designed to hold a single sentence of 40 characters (seq_length) and represent all possible charcters. (creating a fresh character grid from each new character)
        for t, character in enumerate(sentence): # starts a second loop that goes through each character in the current (sentence), keeping track of both characters position within the sentence(t) and the actual character itself (character). its like going through each box on the character grid form.
            x[0, t, char_to_index[character]] = 1 # fills in the (x) structure with a "1" at a specific position that corresponds to the current character. it uses char_to_index. its like ticking the correct character checkbox on the form. 

        predictions = model.predict(x, verbose=0)[0] # the (model.predict) function is used to get the models prediction fora given input data (x). (verbose=0) means that prediction process will be silent without showing progress info. ([0]) at the end extracts the first predictions from the result. (asking a model to predict next char)
        next_index = sample(predictions, temperature) # passes the models (predictions) to the sample function. the sample function adds a controlled level of randomness to the decision-making process, based on the tempreature setting. (deciding on the next character with a touch of randomness)
        next_character = index_to_char[next_index] # uses the (index_to_char) dict to look up the actual character that corresponds to the (next_index). this lines converts the index of the next character back into the actual character using the dict.

        generated += next_character # appends the newly genrated text (next_character) to the (generated) string, effectively adding it to the end of the generated text. 
        sentence = sentence[1:] + next_character # removing the first character of the curent statement (sentence[1:]), making room for the new  character. adding the (next_charcter) to the end of the statement. (the sentence is updated for next iteration)
    return generated # returns the generated text


print ('---------0.2---------')
print (generate_text(300, 0.2)) # 300 characters with randomness of 0.2 temprature. 
