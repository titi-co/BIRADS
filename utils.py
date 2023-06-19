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
from sklearn.model_selection import train_test_split


def loadModel():
    """
    Carrega o modelo a partir do arquivo `model.joblib` e o retorna.

    :return: O modelo carregado.
    """
    return load('model.joblib')


def saveModel(model):
    """
    Salva o modelo em um arquivo chamado `model.joblib`.

    :param model: O modelo a ser salvo.
    """
    dump(model, 'model.joblib')


def glcmEntropy(glcm):
    """
    Recebe uma matriz 4D de GLCMs e retorna uma lista 1D com a entropia de cada GLCM.

    :param glcm: A matriz de coocorrência de níveis de cinza (GLCM).
    :return: A entropia de cada matriz GLCM.
    """
    entropy = []
    for i in range(0, 5):
        for j in range(0, 4):
            entropy.append(shannon_entropy(glcm[:, :, i, j]))
    return entropy


def showFeatures(image, screen):
    """
    Recebe uma imagem e exibe a energia, homogeneidade e entropia da imagem.

    :param image: A imagem a ser analisada.
    :param screen: O objeto de tela que contém a imagem.
    """

    numpyImage = np.array(image, dtype=np.uint8)
    glcm = graycomatrix(numpyImage, distances=[
        1, 2, 4, 8, 16], angles=[0, pi/4, pi/2, 3*pi/4], levels=256)
    energy = graycoprops(glcm, 'energy')
    homogeneity = graycoprops(glcm, 'homogeneity')
    entropy = glcmEntropy(glcm)


    info = ""
    info += f"\nEnergia: {round(energy[0][0],2)}\n"
    info += f"Homogeneidade: {round(homogeneity[0][0],2)}\n"
    info += f"Entropia: {round(entropy[0],2)}"
    screen.text.set(info)


def computeFeatures(image):
    """
    Recebe uma imagem, converte para uma matriz numpy, calcula a matriz de coocorrência de níveis de cinza (GLCM)
    e em seguida, calcula o contraste, homogeneidade e entropia da imagem.

    :param image: A imagem a ser analisada.
    :return: Uma lista contendo o contraste, homogeneidade e entropia da imagem.
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
    Recebe uma imagem, redimensiona para três tamanhos diferentes e em seguida calcula as características de cada
    um desses tamanhos.

    :param image: A imagem a ser processada.
    :return: Uma lista contendo as características.
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
    Recebe uma imagem, equaliza, converte para escala de cinza, quantiza em 16 e 32 cores e em seguida,
    retorna as características de cada imagem quantizada.

    :param path: O caminho para o arquivo de imagem.
    :param image: A imagem a ser processada.
    :return: Uma lista contendo as características.
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


def featuresFolder(screen):
    """
    Lê cada imagem da pasta "train/", e em seguida, chama a função featuresFile para obter as características de cada imagem.

    :param screen: O objeto de tela da interface gráfica.
    :return: Uma lista de características e uma lista de tipos.
    """
    url = "train/"
    types = []
    features = []
    imgCount = 0

    # Lê cada imagem da pasta "train/" e extrai as características
    for i in range(1, 5):
        for file in os.scandir(f"{url}{i}/"):
            if file.path.endswith(".jpg") and file.is_file():
                features.append(featuresFile(file.path))
                types.append(i)
                imgCount += 1
                screen.progressBar['value'] = int((imgCount/5642)*100)
                screen.update_idletasks()
                screen.text.set(f"Gerando características...{imgCount}/5642")
    return features, types


def trainModel(screen):
    """
    Recebe um objeto de tela como argumento e retorna um modelo treinado.

    :param screen: A tela onde o treinamento está acontecendo.
    :return: O classificador treinado.
    """
    start = time()
    X, y = featuresFolder(screen)
    screen.progressBar.destroy()
    X_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=.25)

    classifier = svm.SVC(
        kernel="linear", C=0.01)
    screen.text.set("Treinando...")
    classifier.fit(X_train, y_train)

    y_prediction = classifier.predict(x_test)
    accuracy = model_selection.cross_val_score(classifier, X, y, cv=10)

    confusion = confusion_matrix(y_test, y_prediction)

    end = time()

    totalTime = end - start

    info = ""

    info += f"{confusion}\n"
    info += f"\nAcurácia: {round(accuracy.mean(), 2)}"
    info += f"\nTempo: {round(totalTime, 2)}"
    screen.text.set(info)

    return classifier
