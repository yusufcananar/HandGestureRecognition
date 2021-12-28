# import the necessary packages
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model

# CONSTANTS
classes = ["rock(0)", "paper(1)", "scissors(2)", "OK(3)"]

# Read .csv datas
path = "Resources/"  # Path to .csv datas
rock_dataset = pd.read_csv(path + "0.csv", header=None)  # class = 0 Rock
scissors_dataset = pd.read_csv(path + "1.csv", header=None)  # class = 1 Paper
paper_dataset = pd.read_csv(path + "2.csv", header=None)  # class = 2 Scissors
ok_dataset = pd.read_csv(path + "3.csv", header=None)  # class = 3 OK

# Concatenate all the data
frames = [rock_dataset, scissors_dataset, paper_dataset, ok_dataset]
dataset = pd.concat(frames)

def plot_8sensors_data(data,
                       title,
                       interval=40,
                       no_of_sensors=8,
                       n_steps=8):
    """Plotting 8 sensors 8 sequnetial data in one figure"""

    xTime = np.linspace(interval, interval * n_steps, n_steps)
    yInterval = np.linspace(-128, 128, 1)

    n = 1
    fig = plt.figure()
    for i in range(0, len(data)-1, n_steps):
        plt.subplot(int(no_of_sensors/2), 2, n)
        plt.plot(xTime, data[i: i + n_steps])
        plt.title("{}. sensor".format(n))
        plt.xlabel("time(ms)")
        plt.ylabel("Samples")
        plt.xticks(xTime)
        plt.yticks(yInterval)
        n += 1

    plt.suptitle(title)
    plt.tight_layout()
    # fig.savefig("Resources/" + title + ".png", dpi=100)
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Spectral):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def check_train_test_distribution(y_train,
                                  y_test,
                                  classes,
                                  sequential=False):
    """Observe the distribution of the train and test sets"""
    if sequential:
        y_test = decoder(y_test)
        y_train = decoder(y_train)

    else:
        pass

    train_class_counts = [(y_train == 0).sum(),
                          (y_train == 1).sum(),
                          (y_train == 2).sum(),
                          (y_train == 3).sum()]

    test_class_counts = [(y_test == 0).sum(),
                         (y_test == 1).sum(),
                         (y_test == 2).sum(),
                         (y_test == 3).sum()]

    width_x = max(len(x) for x in classes)
    res = "\n".join(
        "{:>{}} : {}  {}".format(x, width_x, y, z) for x, y, z in zip(classes, train_class_counts, test_class_counts))
    print(res + "\n")


def random_forest_classification(dataset):
    """Random Forest Classification"""

    # Split data into training and test set
    X = dataset.iloc[:, 0:64]
    y = dataset.iloc[:, 64]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    check_train_test_distribution(y_train, y_test, classes)

    # Create a Gaussian Classifier
    RFclf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets
    print("Randomforest Training Session has begun..\n")
    RFclf.fit(X_train, y_train)

    # Predict from Test set
    y_pred = RFclf.predict(X_test)

    # Compare prediction and actual class with accuracy score
    print("Accuracy : {accuracy:.3f}\n".format(accuracy=metrics.accuracy_score(y_pred, y_test)))
    cm = metrics.confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    plot_confusion_matrix(cm, classes=classes, title="Confusion Matrix: RandomForest Model")
    # fig.savefig("Resources/cmRF.png", dpi=100)
    plt.show()


def decoder(y_list):
    """One-hot Decoder Specified for LSTM Classification"""

    y_classes = []
    for el in y_list:
        y_classes.append(np.argmax(el))
    return np.array(y_classes)


def lstm_model(n_steps, n_features):
    """LSTM Model Definition"""
    model = Sequential()
    model.add(
        LSTM(50, return_sequences=True, input_shape=(n_steps, n_features)))

    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=64))
    model.add(Dense(units=128))

    model.add(Dense(units=4, activation="softmax"))
    model.compile(optimizer='adam', loss='mse')

    return model


def lstm_classification(dataset, saved_model=False):
    """LSTM classification"""
    model = None
    history = None

    # sc = MinMaxScaler(feature_range=(0, 1))
    ssc = StandardScaler()
    # Reshape the input data for LSTM net
    X = np.array(dataset.iloc[:, 0:64])
    y = np.array(dataset.iloc[:, 64])

    # Reshape to one flatten vector
    X = X.reshape(X.shape[0] * X.shape[1], 1)
    X = ssc.fit_transform(X)

    X = X.reshape((-1, 8, 8))

    for i in range(len(X)):
        X[i] = X[i].reshape((8, 8)).T

    # Convert to one hot
    y = np.eye(np.max(y) + 1)[y]

    # Split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    check_train_test_distribution(y_train, y_test, classes, sequential=True)

    if saved_model:
        # Load Model
        print("Trained LSTM Model is loading..\n")
        model = load_model('Resources/model_cross_splited_data_SC_250epochs.h5')
        model.summary()

    elif not saved_model:
        # Define the Model
        model = lstm_model(X_train.shape[1], X_train.shape[2]) # n_steps = 8, n_features = 8
        print("LSTM Training Session has begun..\n")
        history = model.fit(X_train, y_train, epochs=250, batch_size=32, verbose=2)

        # list all data in history
        print(history.history.keys())
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # Save
        model.save("Resources/model_cross_splited_data_SC_250epochsv2.h5")
        print("Saved model to disk\n")

    y_pred = model.predict(X_test)

    y_pred_classes = decoder(y_pred)
    y_test_classes = decoder(y_test)

    print("Accuracy : {accuracy:.3f}\n".format(accuracy=metrics.accuracy_score(y_pred_classes, y_test_classes)))
    cm = metrics.confusion_matrix(y_test_classes, y_pred_classes)
    fig = plt.figure()
    plot_confusion_matrix(cm, classes=classes, title="Confusion Matrix: LSTM model")
    # fig.savefig("Resources/nonSCcmLSTM.png", dpi=100)
    plt.show()

if __name__ == '__main__':
    print("Main Function is running..\n")

    plot_8sensors_data(rock_dataset.iloc[1], classes[0])
    plot_8sensors_data(paper_dataset.iloc[1], classes[1])
    plot_8sensors_data(scissors_dataset.iloc[1], classes[2])
    plot_8sensors_data(ok_dataset.iloc[1], classes[3])

    random_forest_classification(dataset)

    lstm_classification(dataset, saved_model=True)




