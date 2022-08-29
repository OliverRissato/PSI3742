import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf; import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
import numpy as np; import sys; from sys import argv

# Use os comandos abaixo se for chamar do prompt
if (len(argv)!=2):
    print("classif1.py nomeimg.ext");
    sys.exit();


from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input, decode_predictions
model = EfficientNetB7(weights='imagenet')
target_size = (600, 600)
img_path = argv[1] #Escreva aqui o diretorio e nome da imagem
img = image.load_img(img_path, target_size=target_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
p=decode_predictions(preds, top=3)[0]


for i in range(len(p)):
    print("%8.2f%% %s"%(100*p[i][2],p[i][1]))