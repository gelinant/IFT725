# Code adapté de projets académiques de la professeur Fei Fei Li et de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Version finale rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin

import numpy as np


def softmax_naive_loss_function(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Calculez la perte softmax (entropie croisée) moyenne et son         #
    #  gradient moyen avc des boucles explicites sur chaque paire (X[i], y[i]). #
    #  N'oubliez pas que l'entropie-croisée pour une paire (X[i], y[i]) est     #
    #  -log(SM[y[i]), où SM est le vecteur softmax à 10 classes de X[i]         #
    #  Pour ce qui est du gradient, vous pouvez utiliser l'equation 4.109       #
    #  du livre de Bishop.                                                      #
    # Stockez la perte dans la variable "loss" et le gradient dans "dW".        #
    # N'oubliez pas la régularisation! Afin d'éviter toute instabilité          #
    # numérique, soustrayez le score maximum de la classe de tous les scores    #
    # d'un échantillon.                                                         #
    #############################################################################
    loss = loss*0
    #dW = dW*0

    for i in range(X.shape[0]):
      # On calcule d'abbord le w^T.X pour l'ensemble de donnée en cours de traitement
      # On un vexteur en sortie de taille C (le nombre de classes)
      predict = np.dot(np.transpose(W), X[i])
      C = predict.size

      SM = np.array(C)
      for j in range(C):
        predict_SM = np.exp(predict[j])/np.sum(np.exp(predict))
        SM = np.append(SM , predict_SM)

        # W et dW ont D dim en sortie donc on multiplie par le X en cours d'analyse pour que ca donne le bon 
        dW[:,j] += (predict_SM - y[i])*X[i]

      #SM = SM - np.max(SM)

      # On a maintenant calculer la loss de la donnée étudiée
      # On va ensuite chercher la bonne valeur theorique pour cette donnée avec y[i]
      # Comme c'est la seule qui sera non nulle, 
      # on va chercher directement le log de la valeur de la classe qui aurait du etre trouvée
      loss += - np.log(SM[y[i]]) + reg 

    loss = loss / X.shape[0]
    dW = dW / X.shape[0]

    #############################################################################
    #                         FIN DE VOTRE CODE                                 #
    #############################################################################

    return loss, dW


def softmax_vectorized_loss_function(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Calculez la perte softmax et son gradient en n'utilisant aucune     #
    #  boucle explicite.                                                        #
    # Stockez la perte dans la variable "loss" et le gradient dans "dW".        #
    # N'oubliez pas la régularisation! Afin d'éviter toute instabilité          #
    # numérique, soustrayez le score maximum de la classe de tous les scores    #
    # d'un échantillon.                                                         #
    #############################################################################
    loss = loss * 0
    dW = dW * 0

    predict = np.dot(X,W)
    SM = np.exp(predict) / np.sum(np.exp(predict),axis=0)
    loss = np.sum(- np.log(SM[y])) + reg 
    loss = loss / X.shape[0]

    dW = np.transpose(np.dot(np.transpose(SM) - y,X))


    print(loss)



    #############################################################################
    #                         FIN DE VOTRE CODE                                 #
    #############################################################################

    return loss, dW
