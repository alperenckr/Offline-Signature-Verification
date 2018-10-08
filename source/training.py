import os
import re
import matplotlib.pyplot as plt
from scipy.misc import imread, toimage
from preprocess.normalize import preprocess_signature
import signet
from cnn_model import CNNModel
import numpy as np
import time
import random
import theano
import theano.tensor as T
import lasagne


os.environ['THEANO_FLAGS'] = "device=cuda,force_device=True,floatX=float32"

canvas_size = (730, 1042)  # Maximum signature size

num_epochs = 60
individuals = {}

plt.ion()
def showChart(epoch, t, v, a):

    #new figure
    plt.figure(0)
    plt.clf()

    #x-Axis = epoch
    e = range(0, epoch)

    #loss subplot
    plt.subplot(211)
    plt.plot(e, train_loss, 'r-', label='Train Loss')
    plt.plot(e, val_loss, 'b-', label='Val Loss')
    plt.ylabel('loss')

    #show labels
    plt.legend(loc='upper right', shadow=True)

    #accuracy subplot
    plt.subplot(212)
    plt.plot(e, val_accuracy, 'g-')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    #show
    plt.show()
    plt.pause(0.5)

def get_dataset(path: str, prep_path: str):
    for dirpath, dirs, _ in os.walk(path):
        for directory in dirs:
            for _, _, files in os.walk(os.path.join(dirpath, directory)):
                for filename in files:
                    fname = os.path.join(dirpath, directory, filename)
                    processed = preprocess_signature(imread(fname, flatten=1), canvas_size)
                    x = re.findall('\\d+', fname)
                    y = filename[0]

                    if not os.path.exists(os.path.join(prep_path, r'' + y + ' (' + x[0] + ')')):
                        os.makedirs(os.path.join(prep_path, r'' + y + ' (' + x[0] + ')'))
                    toimage(processed, cmin=0.0, cmax=...).save(
                        os.path.join(prep_path, r'' + y + ' (' + x[0] + ')') + '/' + x[1] + '.png')

def get_prep_dataset(path: str):
    for dirpath, dirs, _ in os.walk(path):
        for directory in dirs:
            for _, _, files in os.walk(os.path.join(dirpath, directory)):
                for filename in files:
                    fname = os.path.join(dirpath, directory, filename)
                    processed = imread(fname, flatten=1)
                    a = ''.join(
                        [n for n in (os.path.basename(os.path.abspath(os.path.join(fname, '..')))) if n.isdigit()])
                    if int(a) - 1 not in individuals:
                        individuals[int(a) - 1] = []
                    is_forgery = 0
                    if (os.path.basename(os.path.abspath(os.path.join(fname, '..'))))[0] == "F":
                        is_forgery = 1
                    individuals[int(a) - 1].append({'sign': [processed], 'type': is_forgery})
    return


def get_dataset_randomly():
    keys = list(individuals.keys())
    random.shuffle(keys)
    train = keys[0:int(len(keys) * 0.91)]
    test = keys[int(len(keys) * 0.91): len(keys)]
    return train, test


def prepare_dataset(train: list, test: list):
    train_dataset = {}
    val_dataset = {}
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []
    for it in train:
        forgery = 0
        genuine = 0
        train_dataset[it] = []
        val_dataset[it] = []
        for sign in individuals[it]:
            if sign['type'] is 1 and forgery is not 4:
                train_dataset[it].append(sign)
                forgery = forgery + 1
            elif sign['type'] is 1:
                val_dataset[it].append(sign)
            elif sign['type'] is 0 and genuine is not 21:
                train_dataset[it].append(sign)
                genuine = genuine + 1
            else:
                val_dataset[it].append(sign)
    for it in train:
        for sign in train_dataset[it]:
            y_train.append([sign['type'], [0] * 104])
            x_train.append(sign['sign'])
            y_train[-1][1][train.index(it)] = 1
        for sign in val_dataset[it]:
            y_val.append([sign['type'], [0] * 104])
            x_val.append(sign['sign'])
            y_val[-1][1][train.index(it)] = 1

    for it in test:
        for signs in individuals[it]:
            y_test.append([signs['type'], [0] * 104])
            x_test.append(signs['sign'])
    return x_train, y_train, x_val, y_val, x_test, y_test


prep_path = "signatures/UTSig/prepDataset"
path = "signatures/CEDAR"
#get_dataset(path, prep_path)

# get_dataset(path,prep_path)
print("Loading...")
get_prep_dataset(prep_path)
train, test = get_dataset_randomly()
x_train, y_train, x_val, y_val, x_test, y_test = prepare_dataset(train, test)

print("Loaded: " + str(np.array(x_train).shape) + " images")
# input_shape = T.tensor4('shapes')
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
target_var115 = T.imatrix('targets')

model_weight_path = 'models/training_weights.pkl'
from six.moves import cPickle

with open(model_weight_path, 'wb') as f:
    data = {'input_size': (150, 220), 'img_size': (170, 242), 'params': None, 'test': test}
    cPickle.dump(data, f)
model = CNNModel(signet, model_weight_path, input_var)

prediction1, prediction115 = lasagne.layers.get_output([model.model["out"], model.model["out2"]])

loss1 = lasagne.objectives.binary_crossentropy(prediction1, target_var)
loss115 = lasagne.objectives.categorical_crossentropy(prediction115, target_var115)

lambda_val = 0.99

loss = loss1 * lambda_val + loss115 * (1 - lambda_val) * (1 - target_var)
loss = loss.mean()
loss += 0.0001 * lasagne.regularization.regularize_network_params([model.model["out"], model.model["out2"]],
                                                                  lasagne.regularization.l2)
params = lasagne.layers.get_all_params([model.model["out"], model.model["out2"]], trainable=True)

updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.0001)


def isclose(a, b):
    return abs(a - b) <= 0.5


test_prediction1, test_prediction115 = lasagne.layers.get_output([model.model["out"], model.model["out2"]],
                                                                 deterministic=True)
test_loss1 = lasagne.objectives.binary_crossentropy(test_prediction1, target_var)
test_loss115 = lasagne.objectives.categorical_crossentropy(test_prediction115, target_var115)

test_loss = test_loss1 * lambda_val + test_loss115 * (1 - lambda_val) * (1 - target_var)
test_loss = test_loss.mean()
test_acc_1 = T.mean(isclose(test_prediction1, target_var))
test_acc_115 = T.mean(T.eq(T.argmax(test_prediction115, axis=1), T.argmax(target_var115, axis=1)),
                      dtype=theano.config.floatX)

train_fn = theano.function([input_var, target_var, target_var115], loss, updates=updates)
val_fn = theano.function([input_var, target_var, target_var115], [test_loss, test_acc_1, test_acc_115])

print("Starting training...")


# We iterate over epochs:
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

train_loss = []
val_loss = []
val_accuracy = []
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(x_train, y_train, 32, shuffle=False):
        inputs, targets = batch
        target1, target2 = zip(*targets)
        train_err_x = train_fn(inputs, target1, target2)
        train_err += train_err_x
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc1 = 0
    val_acc115 = 0
    val_batches = 0
    for batch in iterate_minibatches(x_val, y_val, 32, shuffle=False):
        inputs, targets = batch
        target1, target2 = zip(*targets)
        err, acc1, acc115 = val_fn(inputs, target1, target2)
        val_err += err
        val_acc1 += acc1
        val_acc115 += acc115
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %  \t\t{:.2f} %".format(
        val_acc1 / val_batches * 100,
        val_acc115 / val_batches * 100
    ))
    train_loss.append(train_err / train_batches)
    val_loss.append(val_err / val_batches)
    val_accuracy.append(val_acc1 / val_batches * 100)
    #showChart(epoch, train_loss, val_loss, val_accuracy)

# After training, we compute and print the test error:
test_err = 0
test_acc = 0
test_acc115 = 0
test_batches = 0
for batch in iterate_minibatches(x_test, y_test, 32, shuffle=False):
    inputs, targets = batch
    target1, target2 = zip(*targets)
    err, acc, acc115 = val_fn(inputs, target1, target2)
    test_err += err
    test_acc += acc
    test_acc115 += acc115
    test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
print("  test accuracy multi:\t\t{:.2f} %".format(test_acc115 / test_batches * 100))

with open(model_weight_path, 'wb') as f:
    x = lasagne.layers.get_all_param_values(model.model["fc2"])
    data = {'input_size': (150, 220), 'img_size': (170, 242), 'params': x, 'test': test}
    cPickle.dump(data, f)
    # original = imread(fname, flatten=1)
    # processed = preprocess_signature(original, canvas_size)
    # print(processed)
