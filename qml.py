import pennylane as qml
from pennylane import numpy as np
import math

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
    qml.CZ(wires=[0,1])
    qml.Rot(param[0],param[1],param[2],wires=[0])
    qml.Rot(param[3],param[4],param[5],wires=[1])

# this should be the full circuit then
@qml.qnode(dev)
def circuit(input,params):
    U_phi(input)

    for param in params:
        W_theta_part(param)

    return [qml.expval(qml.PauliZ(0)),qml.expval(qml.PauliZ(1))]


def data_set_mapping(x,y):
    x = math.sin(x * math.pi * 8)
    y = math.cos(y * math.pi * 8)
    val = (x + y) / 2;
    return 1 if val > 0 else -1


X_data = np.random.random((100,2))
Y_data = [data_set_mapping(i[0],i[1]) for i in X_data]
print(Y_data)

def loss(labels,predictions):
    loss = 0
    for l, p in zip(labels,predictions):
        loss = loss + (l - sum(p)) ** 2
    loss = loss / len(labels)
    return loss

def circ(X,params):
    X = np.append(X,(math.pi - X[0])*(math.pi - X[1]));
    bias = params[0][0]
    return circuit(X,params[1:]) + bias

def cost(params,X,y):
    X = X * math.pi;
    preds = [circ(x,params) for x in X]
    ls = loss(y,preds)
    print(ls)
    return ls

opt = qml.AdamOptimizer(0.01)
params = np.random.random((L+1,6)) * 2 - 1
print(params)


print("---- circuit values ------")
print("should be different?")
print(circ([0,math.pi],params));
print(circ([0,0],params));
print(circ([math.pi,0],params));

for i in range(10):
    params = opt.step(lambda v: cost(v,X_data,Y_data),params)
    print(params)
