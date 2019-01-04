import cv2
import numpy as np
import os
import math
import glob
from sklearn.metrics.pairwise import cosine_similarity

categorias = ["blink", "gru", "pikachu", "mono", "drake", "lisa", "outstanding move", "spiderman"]

#Hay que definir que descriptores vamos a usar, SIFT + que?
def main():
    #primero, nos gustaría hacer un gran corpus con todos los memes
    images = []
    for i in range(len(categorias)):
        path = "memes/" + categorias[i] + "/"
        for file in os.listdir(path):
            images.append([cv2.cvtColor(cv2.imread(os.path.join(path, file)), cv2.COLOR_BGR2GRAY), i, file])
    for tupla in images:
        tupla[0] = cv2.resize(tupla[0], (128, 128))
    query = images[0]
    target = images[5]
    print(query[0].shape)
    print(target[0].shape)
    print("comparando " + target[2] + " de la categoria " + categorias[target[1]])
    #sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(query[0], None)
    kp2, des2 = sift.detectAndCompute(target[0], None)
    #des1 = des1.flatten()
    #des2 = des2.flatten()
    #print(len(des1))
    #print(len(des2))
    #sim = cosine_similarity([des1, des2])
    #print(sim)
    #kp = sift.detect(query[0], None)
    #print(type(kp))
    #bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    #matches = bf.match(des1,des2)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good_points = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            good_points.append(m)
    number_keypoints = 0
    if len(kp1) <= len(kp2):
        number_keypoints = len(kp1)
    else:
        number_keypoints = len(kp2)
    print("Keypoints 1ST Image: " + str(len(kp1)))
    print("Keypoints 2ND Image: " + str(len(kp2)))
    print("GOOD Matches:", len(good_points))
    print("How good it's the match: ", len(good_points) / number_keypoints * 100)
    #print(des1)
    #print(des1.shape)
    #print(len(matches))
    #for m in matches:
        #print(m.distance)
    #print(matches)

main()
