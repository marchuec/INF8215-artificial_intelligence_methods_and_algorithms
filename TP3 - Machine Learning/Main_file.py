###################################################################
## Classificateur d'adresses web selon "phishing" ou "legitimate"
##
## Auteurs :
##      Marc-Antoine Huet (xxxx)
##      Maxim Pecherskiy (xxx)
##      Didier Blach-Laflèche (xxx)
## Date : décembre 2020
##################################################################

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GaussianNoise, BatchNormalization
from tensorflow.keras import Sequential, regularizers, callbacks
from keras.models import model_from_json, model_from_yaml
import pandas as pd
from sklearn.preprocessing import label_binarize, MinMaxScaler, StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import copy
import os

def main():
    # New training or load previous model (TRUE or FALSE)
    new_training = False

    # Data pre-processing
    X_data, y_data, X_test, y_test, X_test_kaggle = dataPreProcessing(random_state=None, process="Std",
                                                                      valid_ensemble_size=0.20)

    if new_training == True:
        # Create models with different training sets
        ensemble, indv_accuracies = fit_data(group_size=5, X_data=X_data, y_data=y_data, validation_set_size=0.10, random_state=None)

        # Performance of the model
        indv_accuracies_new, global_accuracy_cumul = predict_validation_set(ensemble, X_test, y_test)

        #** indv_accuracy = Accuracy of each member on his random validation set
        #** indv_accuracy_new = Accuracy of each member on the new validation set

        # Print performance indicators
        print("\n******************************************************************")
        print("Accuracy des modèles sur les sets random: \n")
        for i, data in enumerate(indv_accuracies):
            print("Group %d - acc = %.6f" % (i, data))
        print("Accuracy moyenne: %.5f" % np.mean(indv_accuracies))

        print("\nAccuracy des modèles sur le new_validation_set: ")
        for i, data in enumerate(indv_accuracies_new):
            print("%.6f" % (data))
        print("\nAccuracy des modèles sur le new_validation_set selon le # de membres: ")
        for i, data in enumerate(global_accuracy_cumul):
            print("%.6f" % (data))
    else:
        global_accuracy_cumul=[0]
        nb_groups = 1
        ensemble = list()
        for group_ite in range(nb_groups):
            model = tf.keras.models.load_model("SavedModels/model_" + str(group_ite) + "weights.h5")
            ensemble.append(model)

    # Generate submission file
    predict_test_set(ensemble, X_test_kaggle, global_accuracy_cumul[-1])

    return None


def dataPreProcessing(random_state, process, valid_ensemble_size):
    """
    Import data from files and pre-process the data (without spliting).

    :param random_state: seed for raw data shuffle
    :param process: type of normalisation ("Norm", "Std" or "None")
    :param performance_test: create third set to evaluate the ensemble
    :return: processed data
    """

    # Data import
    data_set = pd.read_csv("train.csv")
    test_set_kaggle = pd.read_csv("test.csv")
    data_set.drop(["url"], axis=1, inplace=True)
    test_set_kaggle.drop(["url"], axis=1, inplace=True)

    # Data shuffle
    data_set = data_set.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Pandas -> Numpy
    X_data = data_set.values[:,:-1]
    y_data = data_set.values[:,-1]
    X_test_kaggle = test_set_kaggle.values

    # Label encoding
    for i, label in enumerate(y_data):
        if label == "phishing":
            y_data[i] = 0
        elif label == "legitimate":
            y_data[i] = 1
    y_data = y_data.astype("int")
    X_data = X_data.astype("int")
    y_data = np.reshape(y_data, (len(y_data), 1))   # To have vector column

    # Third set to test the ensembleof models
    X_data_ensemble, X_test, y_data_ensemble, y_test = train_test_split(X_data, y_data, test_size=valid_ensemble_size)

    # Data normalization/standardization
    if process != "None":
        if process == "Norm":
            scaler = MinMaxScaler().fit(X_data_ensemble)
        elif process == "Std":
            scaler = StandardScaler().fit(X_data_ensemble)
        X_data_ensemble = scaler.transform(X_data_ensemble)
        X_test = scaler.transform(X_test)
        X_test_kaggle = scaler.transform(X_test_kaggle)

    print("X_data size: (%d,%d)" % X_data.shape)
    print("y_data size: (%d,%d)" % y_data.shape)
    print("X_test size: (%d,%d)" % X_test_kaggle.shape)

    return X_data_ensemble, y_data_ensemble, X_test, y_test, X_test_kaggle


def fit_data(group_size, X_data, y_data, validation_set_size, random_state):
    """
    Fit model on different training and validation sets.

    :param group_size: Number of groups to fit data onto
    :param random_state: seed for data splitting into Training and Validation set
    :return:
    """
    ensemble, indv_accuracy = list(), list()
    for group_ite in range(group_size):
        # Split train set and validation set randomly
        #X_train, X_validation, y_train, y_validation = train_test_split(X_data, y_data, test_size=validation_set_size, random_state=random_state)
        ix = [i for i in range(len(X_data))]
        train_ix = resample(ix, replace=True, n_samples=6750)
        validation_ix = [x for x in ix if x not in train_ix]
        # select data
        X_train, y_train = X_data[train_ix], y_data[train_ix]
        X_validation, y_validation = X_data[validation_ix], y_data[validation_ix]

        model, accuracy = fit_one_model(X_train, y_train, X_validation, y_validation, group_ite)
        ensemble.append(model)

        # Save model to file
        model.save("SavedModels/model_" + str(group_ite) + "weights.h5")
        print("Saved model to disk")

        indv_accuracy.append(accuracy)       # Score sur le random validation_set

    return ensemble, indv_accuracy


def fit_one_model(X_train, y_train, X_validation, y_validation, group_ite):
    """
    Fit the data on a model.

    :return: The model and the Accuracy (max)
    """

    # Model initialization
    model = Sequential()
    model.add(GaussianNoise(0.13, input_shape=(87,))) #0.13
    model.add(Dense(800, activation="relu", name="layer1")) #800
    model.add(Dropout(0.4)) #0.4
    model.add(GaussianNoise(0.06)) #0.06
    model.add(Dense(80, activation="relu", name="layer2"))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation="relu", name="layer3"))
    model.add(Dense(1, activation="sigmoid", name="layerOutput"))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

    # Checkpoint
    checkpoint_filepath = 'Checkpoint.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=0)

    # Callback to stop at a certain accuracy threshold (UNUSED)
    class MyThresholdCallback(tf.keras.callbacks.Callback):
        def __init__(self, threshold, epoch_min):
            super(MyThresholdCallback, self).__init__()
            self.threshold = threshold
            self.epoch_min = epoch_min

        def on_epoch_end(self, epoch, logs=None):
            val_acc = logs["val_accuracy"]
            if val_acc >= self.threshold:
                print(epoch)
                self.model.stop_training = True
    my_callback = MyThresholdCallback(threshold=0.961, epoch_min = 30)

    # Fit to the training data
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
                    batch_size=1024,
                    epochs=100,
                    verbose=0,
                    callbacks=[model_checkpoint_callback])

    # Accuracy
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    ix = np.argmax(val_accuracy)
    max_val_accuracy = val_accuracy[ix]
    max_accuracy = accuracy[ix]
    print("Group %d - Acc = %.4f" % (group_ite, max_val_accuracy))

    # Accuracy graph
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.hlines(y=0.960, xmin=0, xmax=150, color="red")
    plt.title('Model accuracy for group %d' % group_ite)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(which="both")
    plt.yticks(np.arange(0.93, 1, 0.005))
    axes = plt.gca()
    axes.set_ylim([0.93, 1])
    plt.legend(['Train  ' + str(max_accuracy), 'Validation  ' + str(max_val_accuracy)],
               loc='lower right')
    plt.show()

    # Loss function graph
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss for group %d' % group_ite)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Select checkpoint
    model.load_weights("Checkpoint.hdf5")

    return model, max_val_accuracy


def predict_test_set(ensemble, X_test_kaggle, global_accuracy):
    """
    Create the submission file for the test set.

    :param ensemble: All the models
    :param global_score:
    """


    # Calculate prediction mean of all members (***TO REVISIT***)
    ensemble_predictions = list()
    for model in ensemble:
        prediction = model.predict(X_test_kaggle)
        ensemble_predictions.append(prediction)

    ensemble_predictions_mean = np.mean(ensemble_predictions, axis=0)

    # Labelized the predictions
    predictions_labelized = list()
    for prediction in ensemble_predictions_mean:
        if prediction <= 0.5:
            predictions_labelized.append("phishing")
        else:
            predictions_labelized.append("legitimate")

    # Generate output file
    global_score = "{:.5f}".format(global_accuracy)
    with open("submission_" + str(global_score) + ".csv", mode="w") as submission_file:
        file_writer = csv.writer(submission_file, delimiter=",")
        file_writer.writerow(["idx", "status"])
        for i, prediction in enumerate(predictions_labelized):
            file_writer.writerow([i, prediction])

    return None


def predict_validation_set(ensemble, X_test, y_test):
    """
    Predictions and accuracy of each model of "ensemble" on the new_validation_set.
    """

    # Calculate prediction and accuracy of each model on the new_validation_set
    ensemble_predictions, individual_scores, global_accuracy_cumul = list(), list(), list()
    for model in ensemble:
        ensemble_predictions.append(model.predict(X_test))
        _, individual_score = model.evaluate(X_test, y_test, verbose=0)
        individual_scores.append(individual_score)

        # CALCULER LE SCORE POUR X MEMBRES ICI!
        ensemble_copy = copy.deepcopy(ensemble_predictions)
        ensemble_predictions_mean = np.mean(ensemble_copy, axis=0)

        # Binarization of ensemble predictions
        ensemble_predictions_binarized = list()
        for i, prediction in enumerate(ensemble_predictions_mean):
            if prediction <= 0.5:
                ensemble_predictions_binarized.append(0)  # Phishing
            else:
                ensemble_predictions_binarized.append(1)  # Legitimate

        # Calculate accuracy of the ensemble
        ensemble_predictions_binarized = np.reshape(ensemble_predictions_binarized,
                                                    (len(ensemble_predictions_binarized), 1))
        global_accuracy = np.sum(ensemble_predictions_binarized == y_test) / len(y_test)
        global_accuracy_cumul.append(global_accuracy)

    return individual_scores, global_accuracy_cumul

main()
