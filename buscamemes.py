import cv2
import numpy as np
import os
import math
import glob
import heapq
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
    query = images[4]
    print("La imagen query " + query[2])
    sap = 131
    print(images[sap][2] + " ", categorias[images[sap][1]])
    #Usaremos SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    #par keypoints, descriptor de la query
    kp1, des1 = sift.detectAndCompute(query[0], None)
    #Contador para saber que numero del arreglo le corresponde a que foto
    heap = []
    heapq.heapify(heap)
    cont= 0
    for target in images:
        if target[2] == query[2]:
            continue
        #keypoint, descriptor de la imagen target
        kp2, des2 = sift.detectAndCompute(target[0], None)
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        #todos los matches
        matches = flann.knnMatch(des1, des2, k=2)
        good_points = []
        #Encontramos las correspondencias
        for m, n in matches:
            if m.distance < 0.6*n.distance:
                good_points.append(m)
        number_keypoints = 0
        if len(kp1) <= len(kp2):
            number_keypoints = len(kp1)
        else:
            number_keypoints = len(kp2)
        #Creo que esto no es exactamente similitud, pero lo llamare similitud, hay que echarle un ojo
        #la similitud y el nombre de la imagen correspondiente a la similitud
        similitud = [len(good_points) / number_keypoints * 100, target[2], categorias[target[1]]]
        print("nro " + str(cont) + " " + str(target[1]), similitud[0])
        if len(heap) < 10:
            heapq.heappush(heap, similitud)
        else:
            if heap[0][0] < similitud[0]:
                heapq.heapreplace(heap, similitud)
        cont+=1
    print("El nombre de la imagen de query es:")
    print(query[2])
    print("Pertenece a la clase:")
    print(categorias[query[1]])
    print("Las 10 imagenes mas cercanas son")
    for elem in heap:
        print(elem)

main()
