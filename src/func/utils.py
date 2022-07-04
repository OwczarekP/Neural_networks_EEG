import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np


def normalize_data(input_data: np.ndarray):
    norm = np.linalg.norm(input_data)
    return input_data/norm


def get_train_plot(model_history):
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim(0, 1.1)
    plt.legend(['accuracy', 'val_acc'], loc='lower right')
    plt.show()


def get_test_acc(model, X_test, Y_test):
    pred = model.predict(X_test)
    predict_classes = np.argmax(pred,axis=1)
    expected_classes = np.argmax(Y_test,axis=1)
    print(expected_classes.shape)
    print(predict_classes.shape)
    correct = accuracy_score(expected_classes,predict_classes)
    print(f"Training Accuracy: {correct}")