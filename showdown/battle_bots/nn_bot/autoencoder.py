from showdown.battle import Pokemon
# from tensorflow.keras import layers
# from tensorflow.keras.models import Model, Sequential
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
poke_string = open("data/pokedex.json", "r").read()
y = json.loads(poke_string)
poke_list = list(y.keys())
pokemon_initializations = 50
x_train = []
y_train = []
for poke_string in  poke_list:
    for i in range(50):
        pokemon = Pokemon(poke_string, 100)
        pokemon.hp = np.random.randint(0, 100)
        try:
            vector = pokemon.to_vector().numpy()
            x_train.append(np.array(vector))
            y_train.append(np.array(vector))
        except:
            print("pokemon", poke_string, "to_vector does not work")
encoding_dim = 50
model = Sequential()
model.add(Dense(512, activation = "relu", input_shape=(1157,)))
model.add(Dropout(.1))
model.add(Dense(128, activation = "relu", input_shape=(1157,)))
model.add(Dropout(.1))
model.add(Dense(encoding_dim, activation='relu'))
model.add(Dense(128, activation = "relu", input_shape=(1157,)))
model.add(Dense(512, activation = "relu", input_shape=(1157,)))
model.add(Dense(1157, activation = "relu", input_shape=(1157,)))
#model.add(Dense(1157, activation='sigmoid'))
#autoencoder = Model(input_img, decoded)
opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=opt, loss='mae')

model.fit(np.array(x_train), np.array(y_train),
                epochs=3,
                batch_size=64,
                shuffle=True,
          )

print(x_train[0])
model.predict([x_train[0]])


           #     validation_data=(x_train, y_train))
# this model maps an input to its reconstruction
# autoencoder = Model(input_img, decoded)
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# # autoencoder.fit(x_train, y_train, epochs = 100)
# autoencoder.fit(x_train, y_train,
#                 epochs=50,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_train, y_train))