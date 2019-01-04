import cv2
import numpy as np
import os
import math
import glob

categorias = ["blink", "gru", "pikachu", "mono", "drake", "lisa", "outstanding move", "spiderman"]

#Hay que definir que descriptores vamos a usar, SIFT + que?
def main():
    #primero, nos gustaría hacer un gran corpus con todos los memes
    images = []
    for i in range(len(categorias)):
        path = "memes/" + categorias[i] + "/"
        for file in os.listdir(path):
            images.append((cv2.imread(os.path.join(path, file), 0), i))
main()
