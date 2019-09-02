#DREAM custom implementation
import cv2
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Lambda

# load embeddings

# load yaw values and create pairs of frontal and profile
with open('/home/ganesh/Desktop/yaw_values.txt','r') as myfile:
    data=myfile.read()
    yaw_data_list=list()
    actor_list=list()
    frontal_list=list()
    profile_list=list()
    x=data.strip().split('\n')
    for i in range(len(x)):
        y=x[i]
        z=y.split(':')
        actor=z[0].split('_')[0]
        yaw=z[1]
        
        if abs(yaw)<=20:
            frontal_list.append(z)
        elif abs(yaw)>=45:
            profile_list.append(z)
        actor_list.append(actor)
    myfile.close()
for actors in actor_list:



    

 #defining final yaw calculation
 def yaw_coeff(yaw):
     value=(((4/180)*yaw) -1 )
     sig_value=1./(1.+e^(-value))
     return sig_value
        

# neural network definition
inputs=Input(shape=(128,2,))
output = Dense(128,activation='relu')(inputs)
outputs = Dense(128, activation='relu')(output)
outputs = Dropout(0.4)(outputs)
model=Model(inputs=inputs,outputs=new_embed_calc(outputs,0.5,inputs))

def new_embed_calc(outputs,yaw,inputs):
    yaw_coefficient = yaw_coeff(yaw)
    final_value= inputs + (outputs*yaw_coefficient)
    return final_value
    


# final embedding definition


# custom loss function

# training


