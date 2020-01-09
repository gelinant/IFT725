# Code adapté de projets académiques de la professeur Fei Fei Li et de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Version finale rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin

import numpy as np


def svm_naive_loss_function(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    loss = 0.0
    #############################################################################
    # TODO: Calculez le gradient "dW" et la perte "loss" et stockez le résultat #
    #  dans "dW et dans "loss".                                                 #
    #  Pour cette implementation, vous devez naivement boucler sur chaque pair  #
    #  (X[i],y[i]), déterminer la perte (loss) ainsi que le gradient (voir      #
    #  exemple dans les notes de cours).  La loss ainsi que le gradient doivent #
    #  être par la suite moyennés.  Et, à la fin, n'oubliez pas d'ajouter le    #
    #  terme de régularisation L2 : reg*||w||^2                                 #
    #############################################################################

    #https://ljvmiranda921.github.io/notebook/2017/02/11/multiclass-svm/

    current_loss = 0

    for i in range(X.shape[0]):
        # Produit scalaire entre W et X
        predict = np.dot(np.transpose(W), X[i])
        
        for j in range(predict.size):
            if j != y[i]:
                current_loss = np.max([0, 1 + predict[j] - predict[y[i]]])

                # Loss + terme de regularisation L2
                loss += current_loss + reg * np.linalg.norm(W[i])**2

                # On change la valeur en valeur binaire : 1 si > 0, 0 sinon
                if current_loss != 0:
                    current_loss = 1

                # Gradient
                # dWyi = - I(WjT.Xi - WyiT.Xi + 1 > 0).Xi avec I(...) = 1 si ... > 0
                dW[:,y[i]] -= current_loss * X[i]
                # dWj = I(WjT.Xi - WyiT.Xi + 1 > 0).Xi avec I(...) = 1 si ... > 0
                dW[:,j]    += current_loss * X[i]

    # Moyenne pour l'ensemble des exemples
    loss = loss / X.shape[0]
    dW = dW / X.shape[0]

    #print("loss :", loss)
    #print("dW :", dW)

    #############################################################################
    #                            FIN DE VOTRE CODE                              #
    #############################################################################

    return loss, dW


def svm_vectorized_loss_function(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO: Implémentez une version vectorisée de la fonction de perte SVM.     #
    # Veuillez mettre le résultat dans la variable "loss".                      #
    # NOTE : Cette fonction ne doit contenir aucune boucle                      #
    #############################################################################

    # Produits scalaires entre W et X
    list_predict = np.dot(np.transpose(W), np.transpose(X))
    
    #print(list_predict.shape)
    #print(np.arange(X.shape[0]))

    list_current_loss = 1 + list_predict - list_predict[y, np.arange(X.shape[0])] + reg * np.linalg.norm(W)**2
    
    # On enleve les cas ou j = y[i] comme precedemment
    list_current_loss[y, np.arange(X.shape[0])] = 0
    # On enleve les cas ou c'est inferieur a 0 comme precedemment
    list_current_loss = np.where(list_current_loss > 0, list_current_loss, 0)
    
    # Loss + terme de regularisation L2
    loss = np.sum(list_current_loss)

    # Moyenne pour l'ensemble des exemples
    loss = loss / X.shape[0]

    #############################################################################
    #                            FIN DE VOTRE CODE                              #
    #############################################################################

    #############################################################################
    # TODO: Implémentez une version vectorisée du calcul du gradient de la      #
    #  perte SVM.                                                               #
    # Stockez le résultat dans "dW".                                            #
    #                                                                           #
    # Indice: Au lieu de calculer le gradient à partir de zéro, il peut être    #
    # plus facile de réutiliser certaines des valeurs intermédiaires que vous   #
    # avez utilisées pour calculer la perte.                                    #
    #############################################################################

    # On change la liste en liste binaire : 1 si > 0, 0 sinon
    list_current_loss = np.where(list_current_loss > 0, 1, 0)

    # On recupere le nombre de 1 pour chaque ligne (donnees au dela de la "marge")
    list_nb_ones = np.sum(list_current_loss, axis=0)
    
    # On soustrait la valeur a l'etiquette de la ligne correspondante
    list_current_loss[y, np.arange(X.shape[0])] -= list_nb_ones

    #print(list_current_loss.shape)

    # Gradient
    dW = np.dot(np.transpose(X), np.transpose(list_current_loss))

    # Moyenne pour l'ensemble des exemples
    dW = dW / X.shape[0]

    #############################################################################
    #                            FIN DE VOTRE CODE                              #
    #############################################################################

    return loss, dW
