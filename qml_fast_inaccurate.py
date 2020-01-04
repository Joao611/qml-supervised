import pennylane as qml
from pennylane import numpy as np
import math
from sklearn import metrics
import random
import scipy

random.seed(1)
np.random.seed(1)

print("================== start ==============")

# Circuit is hardcoded in several places to only have 2 qubits, despite the following variables.
# For instance, the feature_map() only takes in 1 value "phi",
# whereas multiple values would be needed for more qubits.
num_qubits = 2
shots = 10
dev = qml.device("default.qubit", wires=num_qubits, shots=shots)

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
    qml.CZ(wires=[1,0])
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
random_unitary = scipy.stats.unitary_group.rvs(4)

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
    # This is not enterely the same as in the paper but the implementation is a lot faster in pennylane this way.
    # And it still seems to work
    data = circuit(X,params)
    return data[0]*data[1]

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
opt = qml.AdamOptimizer(0.01)
# Not sure how the paper initializes parameters
# If i understand some formula that it is initialized as zero
# This is kinda uncommon in ML though
params = np.random.normal(0,1,(L+1, 4)) * 2  - 1
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

batch_size = 10
for i in range(20):
    print("EPOCH: " + str(i))
    print(params)
    params = opt.step(lambda v: cost(v,X_data,Y_data),params)
    evaluate()
