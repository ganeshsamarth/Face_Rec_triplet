#DREAM custom implementation
import cv2
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Lambda

# load embeddings


#defining final yaw calculation
def yaw_coeff(yaw):
    value=(((4/180)*yaw) -1 )
    sig_value=1./(1.+e^(-value))
    return sig_value
def only_first_vector(x):
    return x[1:]

def only_yaw_vector(x):
    return x[0]

def new_embed_calc(outputs,yaw,inputs):
    yaw_coefficient = yaw_coeff(yaw)
    final_value= inputs + (outputs*yaw_coefficient)
    return final_value
    

# neural network definition
inputs=Input(shape=(129,))
layer1 = Lambda(only_first_vector)
yaw_layer = Lambda(only_yaw_vector)
layer2 = layer1(inputs)
output = Dense(128,activation='relu')(layer2)
outputs = Dense(128, activation='relu')(output)
outputs = Dropout(0.4)(outputs)
layer_final = Lambda(new_embed_calc)
final_output = layer_final(outputs,yaw_layer(inputs),layer2)
model=Model(inputs=inputs,outputs=final_output)

# training

model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=[accuracy])

model.fit()


