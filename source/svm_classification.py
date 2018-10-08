import os
import random

import six
from scipy.misc import imread
from sklearn import svm

import signet
from cnn_model import CNNModel
from six.moves import cPickle

canvas_size = (816, 888)  # Maximum signature size

individuals = {}
x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []


def get_prep_dataset(path: str):
    for dirpath, dirs, _ in os.walk(path):
        for directory in dirs:
            for _, _, files in os.walk(os.path.join(dirpath, directory)):
                for filename in files:
                    fname = os.path.join(dirpath, directory, filename)
                    processed = imread(fname, flatten=True)
                    a = ''.join(
                        [n for n in (os.path.basename(os.path.abspath(os.path.join(fname, '..')))) if n.isdigit()])
                    if int(a) - 1 not in individuals:
                        individuals[int(a) - 1] = []
                    is_genuine = 1
                    if (os.path.basename(os.path.abspath(os.path.join(fname, '..'))))[0] == "F":
                        is_genuine = 0
                    individuals[int(a) - 1].append({'sign': processed, 'type': is_genuine})
    return


def get_dataset_randomly():
    keys = list(individuals.keys())
    random.shuffle(keys)
    train = keys[0:int(len(keys) * 0.85)]
    test = keys[int(len(keys) * 0.85): len(keys)]
    return train, test


def prepare_dataset(train: list, test: list):
    x_dataset = []
    y_dataset = []

    for it in train:
        for signs in individuals[it]:
            y_dataset.append([signs['type'], [0] * 115])
            x_dataset.append(signs['sign'])
            y_dataset[-1][1][it] = 1
    dataset = list(zip(x_dataset, y_dataset))
    random.shuffle(dataset)
    train = dataset[0:int(len(dataset) * 0.90)]
    val = dataset[int(len(dataset) * 0.90):len(dataset)]
    x_train, y_train = zip(*train)
    x_val, y_val = zip(*val)

    for it in test:
        for signs in individuals[it]:
            y_test.append([signs['type'], [0] * 115])
            x_test.append(signs['sign'])
            y_test[-1][1][it] = 1
    return x_train, y_train, x_val, y_val, x_test, y_test


prep_path = "signatures/CEDAR_PREP2"
# path = "/home/faruk/tensorflow/sigver_wiwd-master/sigver_wiwd/signatures/UTSig/Dataset"

# print(z)
# get_dataset(path,prep_path)
# print("Loading...")
# get_prep_dataset(prep_path)
# train, test = get_dataset_randomly()
# x_train, y_train, x_val, y_val, x_test, y_test = prepare_dataset(train, test)
"""
print("Loaded: "+str(np.array(x_train).shape)+" images")
canvas_size = (1627, 2422)
print(np.array(x_train[:]).shape)
"""
model_weight_path = 'models/training_weights.pkl'
model = CNNModel(signet, model_weight_path)

with open(model_weight_path, 'rb') as f:
    if six.PY2:
        model_params = cPickle.load(f)
    else:
        model_params = cPickle.load(f, encoding='latin1')

svm_train = {}

accuracy = 0
total_fp = 0
total_fn = 0
min_far_frr = 1000
eer = 0
total_impostor = 0
total_genuine = 0
total_far = 0
total_frr = 0
sign_count_for_training = 15
for i in range(55):
    # os.walk(os.path.join(prep_path,"/F (" + str(i) + ")"))
    directory = r"o (" + str(i + 1) + ")"
    svm_train[i] = []
    for _, _, files in (os.walk(os.path.join(prep_path, directory))):
        for filename in files[:sign_count_for_training]:
            # print(files[:15])
            fname = os.path.join(prep_path, directory, filename)
            svm_train[i].append(imread(fname, flatten=1))

for i in range(55):
    svm_test = []
    directory = r"o (" + str(i + 1) + ")"
    for _, _, files in (os.walk(os.path.join(prep_path, directory))):
        for filename in files[sign_count_for_training:]:
            fname = os.path.join(prep_path, directory, filename)
            # print(files[15:])
            svm_test.append({'sign': imread(fname, flatten=1), 'type': 1})

    directory = r"f (" + str(i + 1) + ")"
    for _, _, files in (os.walk(os.path.join(prep_path, directory))):
        for filename in files:
            fname = os.path.join(prep_path, directory, filename)
            # print(files[15:])
            svm_test.append({'sign': imread(fname, flatten=1), 'type': 0})

            # print(svm_train[])
    clf = svm.SVC(kernel='rbf', C=1.0, gamma=2 ** -11)
    X = []
    y = []
    weight = []
    for j in svm_train:
        # print(np.array(i).shape)
        x = model.get_feature_vector_multiple(svm_train[j])
        # print(np.array(x).shape)
        X.extend(x)
        if j == i:
            y.extend([1] * len(x))
            weight.extend([sign_count_for_training] * len(x))
        else:
            y.extend([0] * len(x))
            weight.extend([sign_count_for_training * (55 - 1)] * len(x))

    clf.fit(X, y, sample_weight=weight)

    true_pred = 0
    fp = 0
    fn = 0
    impostor = 0
    genuine = 0

    for k in svm_test:
        if k["type"] == 0:
            impostor += 1
        else:
            genuine += 1
        prediction = clf.predict(model.get_feature_vector(k['sign']))
        if prediction == k['type']:
            true_pred += 1
        elif k['type'] == 0:
            fp += 1
        else:
            fn += 1
    total_impostor += impostor
    total_genuine += genuine
    far = fp / impostor
    frr = fn / genuine
    if abs(far - frr) < min_far_frr:
        min_far_frr = abs(far - frr)
        eer = (far + frr) / 2
    print("accuracy: %{:.2f}".format(true_pred * 100 / len(svm_test)))
    print("far: %{:.2f}, frr: %{:.2f}".format(far * 100, frr * 100))
    accuracy += true_pred * 100 / len(svm_test)
    total_far += far
    total_frr += frr
    total_fp += fp
    total_fn += fn

print("total accuracy: %{:.2f}".format(accuracy / 55))
print("FAR: %{}, FRR: %{:.2f}".format(total_far * 100 / 55,
                                  total_frr * 100 / 55))
print(eer)
