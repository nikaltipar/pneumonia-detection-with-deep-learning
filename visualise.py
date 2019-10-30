import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tf_explain.core.integrated_gradients import IntegratedGradients
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(real_labels, predicted_labels, dest_filename, title = 'Confusion matrix', cmap = plt.cm.Blues):
    cm = confusion_matrix(real_labels, predicted_labels)
    classes = ['Normal', 'Pneumonia']
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    fig.savefig(dest_filename)

def integratedGradient(model, filename, target_size, destination, num_iterations = 20):
    img = tf.keras.preprocessing.image.load_img(filename, target_size=target_size)
    img = np.array(tf.keras.preprocessing.image.img_to_array(img)) / 255.

    data = ([img], None)

    explainer = IntegratedGradients()
    grid = explainer.explain(data, model, 0, n_steps = num_iterations)
    explainer.save(grid, ".", destination)

def integratedGradients(model, filenames, target_size, num_iterations = 20):
    i = 1
    n = len(filenames)
    for filename in filenames:
        integratedGradient(model, filename, target_size, "temp1.png")
        fig = plt.figure()
        plt.subplot(n, 2, i)
        i += 1
        img = tf.keras.preprocessing.image.load_img(filename, target_size=target_size)
        plt.imshow(img)
        plt.subplot(n, 2, i)
        i += 1
        img = tf.keras.preprocessing.image.load_img("temp1.png", target_size=target_size)
        plt.imshow(img)
        fig.savefig("integrated_grads.png")

def generateHistoryPlots(history, dest_filename = "history.png"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(16,9))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    fig.savefig(dest_filename)
