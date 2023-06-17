from cmath import pi
import os
from time import time
from PIL import Image, ImageOps
from joblib import dump, load
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import numpy as np
from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split


def loadModel():
    """
    Carrega o modelo do arquivo `model.joblib`.
    
    :return: Modelo carregado.
    """
    return load('model.joblib')


def saveModel(model):
    """
    Salva o modelo dado no arquivo `model.joblib`.

    :param model: Modelo para ser salvo.
    """
    dump(model, 'model.joblib')


def glcmEntropy(glcm):
    """
    Recebe um array 4D de GLCMs e retorna um array 1D com a entropia de cada GLCM.

    :param glcm: Matriz de co-ocorrência de nível cinza (GLCM).
    :return: Entropia da matriz GLCM.
    """
    entropy = []
    for i in range(0, 5):
        for j in range(0, 4):
            entropy.append(shannon_entropy(glcm[:, :, i, j]))
    return entropy


def showFeatures(image, screen):
    """
    Recebe uma imagem e retorna a energia, homogeneidade e entropia da imagem.

    :param image: Imagem a ser analisada.
    :param screen: Objeto de tela que contém a imagem.
    """
    start = time()

    numpyImage = np.array(image, dtype=np.uint8)
    glcm = graycomatrix(numpyImage, distances=[
        1, 2, 4, 8, 16], angles=[0, pi/4, pi/2, 3*pi/4], levels=256)
    energy = graycoprops(glcm, 'energy')
    homogeneity = graycoprops(glcm, 'homogeneity')
    entropy = glcmEntropy(glcm)

    end = time()

    info = ""
    info += f"\nEnergy: {round(energy[0][0],2)}\n"
    info += f"Homogeneity: {round(homogeneity[0][0],2)}\n"
    info += f"Entropy: {round(entropy[0],2)}"
    info += f"\nTime: {round(end - start, 2)}"
    screen.text.set(info)


def computeFeatures(image):
    """
    Recebe uma imagem, converte-a para um array numpy, calcula a matriz de co-ocorrência de nível cinza (GLCM),
    e então calcula o contraste, a homogeneidade e a entropia da imagem.

    :param image: Imagem a ser analisada.
    :return: Lista com o contraste, homogeneidade e entropia da imagem.
    """
    numpyImage = np.array(image, dtype=np.uint8)
    glcm = graycomatrix(numpyImage, distances=[
                        1, 2, 4, 8, 16], angles=[0, pi/4, pi/2, 3*pi/4], levels=256)
    contrast = graycoprops(glcm, 'contrast')
    homogeneity = graycoprops(glcm, 'homogeneity')
    entropy = glcmEntropy(glcm)

    numpyContrast = np.hstack(contrast)
    numpyHomogeneity = np.hstack(homogeneity)

    return list(numpyContrast) + list(numpyHomogeneity) + list(entropy)


def featuresSizes(image):
    """
    Recebe uma imagem, redimensiona-a para três tamanhos diferentes e então calcula as características para cada um
    desses tamanhos.

    :param image: Imagem a ser processada.
    :return: Lista de características.
    """
    size128Image = image.resize((128, 128))
    size64Image = image.resize((64, 64))
    size32Image = image.resize((32, 32))

    size128features = computeFeatures(size128Image)
    size64features = computeFeatures(size64Image)
    size32features = computeFeatures(size32Image)

    features = size32features + size64features + size128features

    return features


def featuresFile(path=None, image=None):
    """
    Recebe uma imagem, equaliza-a, converte-a para escala de cinza, quantiza-a em 16 e 32 cores, e
    então retorna as características para cada imagem quantizada.

    :param path: Caminho para o arquivo de imagem.
    :param image: Imagem a ser processada.
    :return: Lista de características.
    """
    if image is None:
        image = Image.open(path)

    equalizedImage = ImageOps.equalize(image)
    grayImage = equalizedImage.convert("L")
    quant16Image = grayImage.quantize(colors=16)
    quant32Image = grayImage.quantize(colors=32)
    features = featuresSizes(
        quant16Image) + featuresSizes(quant32Image)

    return features


def featuresFolder():
    """
    Lê cada imagem da pasta imgs e então chama a função featuresFile para obter as
    características de cada imagem.

    :return: Lista de características e uma lista de tipos.
    """
    url = "imgs/"
    types = []
    features = []
    imgCount = 0

    for i in range(1, 5):
        for file in os.scandir(f"{url}{i}/"):
            if file.path.endswith(".png") and file.is_file():
                features.append(featuresFile(file.path))
                types.append(i)
                imgCount += 1
    return features, types


def computeMetrics(confusion):
    """
    Calcula a sensibilidade média e a especificidade da matriz de confusão.

    :param confusion: Matriz 3x3 onde as linhas são a classe real e as colunas são a classe prevista.
    :return: Sensibilidade média e especificidade.
    """
    meanSensibility = 0
    for i in range(0, 3):
        meanSensibility += confusion[i][i] / 100

    sum = 0

    for i in range(0, 3):
        for j in range(0, 3):
            if i != j:
                sum += confusion[i][j] / 300
    specificity = 1 - sum

    return (meanSensibility, specificity)


def trainModel():
    """
    Treina o modelo.

    :return: Modelo treinado.
    """
    start = time()
    X, y = featuresFolder()
    X_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=.25)

    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [
        1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'linear']}

    grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)
    grid.fit(X_train, y_train)

    print(grid.best_estimator_)


trainModel()
