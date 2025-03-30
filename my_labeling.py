__authors__ = ['1674585', '1673970', '1672969']
__group__ = 'dl-cc'

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval, visualize_k_means, \
    Plot3DCloud
import Kmeans as km
from Kmeans import *
import KNN as knn
from KNN import *
import time
import matplotlib.pyplot as plt
import math


def retrieval_by_color(imatges, etiquetes, colors_buscar):
    imatges_matching = list()
    for i, etiqueta, in enumerate(etiquetes):
        if etiqueta in colors_buscar:
            imatges_matching.append(imatges[i])
    return imatges_matching


def retrieval_by_shape(imatges, etiquetes, shape_buscar):
    imatges_matching = list()
    for i, etiqueta, in enumerate(etiquetes):
        if etiqueta == shape_buscar:
            imatges_matching.append(imatges[i])
    return imatges_matching


def retrieval_combined(imatges, etiquetes_color, etiquetes_forma, preg_color, preg_forma):
    imatges_matching = retrieval_by_color(imatges, etiquetes_color, preg_color)
    imatges_matching = retrieval_by_shape(imatges_matching, etiquetes_forma, preg_forma)
    return imatges_matching


def Kmeans_statistics(Kmeans, Kmax):
    distancias = list()
    tiempos = list()
    iteraciones = list()

    for i in range(2, Kmax):
        Kmeans.K=i
        tiempo_inicial = time.time()
        Kmeans.fit()
        tiempo_final = time.time()
        distancias.append(Kmeans.withinClassDistance())
        tiempos.append(tiempo_final - tiempo_inicial)
        iteraciones.append(Kmeans.num_iter)

    return distancias, tiempos, iteraciones


def get_shape_accuracy(etiquetes, ground_truth):
    counter = 0
    for etiqueta in etiquetes:
        if etiqueta == ground_truth:
            counter += 1
    return (counter / len(ground_truth)) * 100


def get_color_accuracy(kmeans_labels, ground_t):
    correct_colors = 0
    total_colors = 0
    for km_label, gt_label in zip(kmeans_labels, ground_t):
        union_labels = set(km_label).union(set(gt_label))  # Unió entre kmlabels y groundtruth
        intersection_labels = set(km_label).intersection(set(gt_label))  # Intersecció entre kmlabels i groundtruth
        correct_colors += len(intersection_labels) / len(union_labels)  # Divisió entre intersecció i unió
        total_colors += 1
    percent_accuracy = (correct_colors / total_colors) * 100

    return percent_accuracy


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)


    imagenes = test_imgs
    llistaColor = []
    for i in range(0, 150):
        km = KMeans(imgs[i], 2)
        km.find_bestK(10,  'FC') #modifiquem el segon valor segons l'heurística d'estudi
        km.fit()
        llistaColor.append(get_colors(km.centroids))
        Plot3DCloud(km)
        visualize_k_means(km, [80, 60, 3])
