import pennylane as qml
from pennylane import numpy as np
import math
from sklearn import metrics
import matplotlib.pyplot as plt
import random
from scipy import stats

random.seed(1)
np.random.seed(1)

print("================== start ==============")

# Circuit is hardcoded in several places to only have 2 qubits, despite the following variables.
# For instance, the feature_map() only takes in 1 value "phi",
# whereas multiple values would be needed for more qubits.
num_qubits = 2
# We need to manually evalutate the circuit multiple times since autograd does not work with sample
shots = 1

# Analytic needs to be false cause we want exp val to only return -1 or 1
dev = qml.device("default.qubit", wires=num_qubits, shots=shots,analytic=False)

L = 5

# This should be U_phi for 2 inputs
def U_phi(phi):
    qml.Hadamard(0) @ qml.Hadamard(1)
    qml.RZ(phi[0],wires=[0])
    qml.RZ(phi[1],wires=[1])
    qml.CNOT(wires=[0,1])
    qml.RZ(phi[2],wires=[1])
    qml.CNOT(wires=[0,1])

    qml.Hadamard(0) @ qml.Hadamard(1)
    qml.RZ(phi[0],wires=[0])
    qml.RZ(phi[1],wires=[1])
    qml.CNOT(wires=[0,1])
    qml.RZ(phi[2],wires=[1])
    qml.CNOT(wires=[0,1])


# this should be 1 part of W(theta)
# Should both rotations have the same parameters? or different?
# Figure seems to imply same params
def W_theta_part(param):
    # Entangling section
    qml.CZ(wires=[0,1])
    # Variational section
    qml.RY(param[0],wires=0)
    qml.RY(param[1],wires=1)
    qml.RZ(param[2],wires=0)
    qml.RZ(param[3],wires=1)

# this should be the full circuit then
@qml.qnode(dev)
def circuit(input,params):
    U_phi(input)

    #First a layer of rotations before the entangling part
    qml.RY(params[0][0],wires=0)
    qml.RY(params[0][1],wires=1)
    qml.RZ(params[0][2],wires=0)
    qml.RZ(params[0][3],wires=1)
    for param in params[1:]:
        W_theta_part(param)

    return [qml.expval(qml.PauliZ(0)),qml.expval(qml.PauliZ(1))]

#Build the dataset
def gen_special_unitary():
    unit = stats.unitary_group.rvs(4)
    det = np.linalg.det(unit)
    print(det)
    return unit / det ** (1/4)

random_unitary = gen_special_unitary()

@qml.qnode(dev)
def data_set_circuit(input):
    U_phi(input)

    qml.QubitUnitary(random_unitary,wires=[0,1])

    return [qml.expval(qml.PauliZ(0)),qml.expval(qml.PauliZ(1))]

def data_set_mapping(x,y):
    X = np.array([x,y])
    X = X * math.pi * 2
    X = np.append(X,(math.pi - X[0])*(math.pi - X[1]))
    Y = data_set_circuit(X)
    return Y[0]*Y[1]

def build_dataset(size):
    data = []
    neg_count = 0
    pos_count = 0
    while True:
        X = list(np.random.random((2,)))
        Y = data_set_mapping(X[0],X[1])

        # Implement the 0.3 seperation in the dataset
        if Y < -0.30 and neg_count < size:
            neg_count += 1
            data.append([X,-1.0])
        elif Y > 0.30 and pos_count < size:
            pos_count += 1
            data.append([X,1.0])

        # Make sure there are 40 of each label in the set
        if neg_count > size - 1 and pos_count > size - 1:
            break
    X = [val[0] for val in data]
    Y = [val[1] for val in data]
    return np.array(X),np.array(Y)

X_data,Y_data = build_dataset(40)
print(Y_data)


# Calculate loss for labels
# The loss in the paper does not work with outgrad so we use mean square error.
def loss(labels,predictions):
    loss = 0
    for l, p in zip(labels,predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss

R = 5
def circ(X,params):
    X = np.append(X,(math.pi - X[0])*(math.pi - X[1]))
    data = np.array([circuit(X,params) for x in range(R)])
    data = data.transpose()
    a = data[0]
    b = data[1]
    # then apply the parity function
    # Which is this
    bit_string = a * b
    # This is not entirely the same as the paper.
    # The paper just assignes -1 or 1 based on which value was more present in the shots
    # However since pennylane works with autograd we need to have the output of the
    # circuit be directly used in the calculation of the loss
    # So instead we take the mean value since median does not have a derivative implementation
    return np.mean(bit_string)

def cost(params,X,y):
    # Not sure it is nessary, but since a full rotation is 2 pi making
    # the features use the full range seems beneficial
    X = X * math.pi
    preds = [circ(x,params) for x in X]
    ls = loss(y,preds)
    print("LOSS: " + str(ls))
    return ls


# Paper optimizer is not implemented in pennylane so we just use a different one.
# It seems to work file
opt = qml.AdamOptimizer(0.003)
# Not sure how the paper initializes parameters
# If i understand some formula that it is initialized as zero
# This is kinda uncommon in ML though
params = np.random.random((L+1, 4)) * 2  - 1
#params = np.zeros((L+1, 4))
print(params)

# Evalutates the resulting circuit
def evaluate():
    X_test,Y_test = build_dataset(100)
    X_test = np.array(X_test) * math.pi
    preds = np.array([circ(x,params) for x in X_test])
    preds[preds < 0] = 0
    preds[preds > 0] = 1
    Y_test[Y_test < 0] = 0
    Y_test[Y_test > 0] = 1
    print(metrics.classification_report(Y_test,preds))
    print(metrics.accuracy_score(Y_test,preds))

def plot_dataset():
    num_samples = 50.0
    data = [[data_set_mapping(x/num_samples,y/num_samples) for x in range(int(num_samples))] for y in range(int(num_samples))]
    data = np.array(data)
    data[data < -0.15] = -1
    data[data > 0.15] = 1
    data[np.logical_and(data != -1,data != 1)] = 0
    plt.imshow(data,interpolation='nearest') 
    plt.show()

# def plot_classifier():
#     num_samples = 50.0
#     data = [[circ([x/num_samples,y/num_samples],params) for x in range(int(num_samples))] for y in range(int(num_samples))]
#     data = np.array(data)
#     data[data < -0.15] = -1
#     data[data > 0.15] = 1
#     data[np.logical_and(data != -1,data != 1)] = 0
#     plt.imshow(data,interpolation='nearest') 
#     plt.show()

# plot_dataset()
# plot_classifier()
batch_size = 10
for i in range(20):
    print("EPOCH: " + str(i))
    print(params)
    for i in range(len(X_data) // batch_size):
        idx = i * batch_size
        X_batch = X_data[idx:idx+batch_size]
        Y_batch = Y_data[idx:idx+batch_size]
        params = opt.step(lambda v: cost(v,X_batch,Y_batch),params)
    # plot_classifier()
    evaluate()
