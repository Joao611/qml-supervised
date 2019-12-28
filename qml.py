import pennylane as qml
from pennylane import numpy as np
import math

# Circuit is hardcoded in several places to only have 2 qubits, despite the following variables.
# For instance, the feature_map() only takes in 1 value "phi",
# whereas multiple values would be needed for more qubits.
num_qubits = 2
shots = 10
layers = 5

dev = qml.device("default.qubit", wires=num_qubits, shots=shots)

def state_preparation():
	qml.BasisState(np.zeros(num_qubits), wires=[i for i in range(num_qubits)])

def circU(x=None):
	phi = (math.pi - x[0]) * (math.pi - x[1])
	qml.RZ(x[0], wires=0)
	qml.RZ(x[1], wires=1)
	qml.CNOT(wires=[0, 1])
	qml.RZ(phi, wires=1)
	qml.CNOT(wires=[0, 1])

def feature_map(x=None):
	for i in range(num_qubits):
		qml.Hadamard(i)
	circU(x)
	for i in range(num_qubits):
		qml.Hadamard(i)
	circU(x)

def local_rots(layer_weights):
	for i in range(num_qubits):
		qml.Rot(layer_weights[0], layer_weights[1], 0, wires=i)

def entanglement_gate():
	for i in range(num_qubits):
		qml.CZ(wires=[i, (i+1) % num_qubits])

def layer_variational(weights, layer=None):
	entanglement_gate()
	local_rots(weights[layer])

def weights_variational(weights):
	local_rots(weights[0])
	for l in range(layers):
		layer_variational(weights, layer=l+1)

@qml.qnode(dev)
def circuit(weights, x=None):
	state_preparation()
	feature_map(x)
	# weights_variational(weights)
	return [qml.sample(qml.PauliZ(wire)) for wire in range(num_qubits)]

# (layers+1) x qubits
weights = [
	[1, 2],
	[1, 1],
	[1, 1],
	[1, 1],
	[1, 1],
	[1, 1]
]
# phi = 1
for x in [[0, 1], [0.5, 1], [1, 0], [0.1, 0.2]]:
	print("x = ", x)
	print(circuit(weights, x=x))
	print(circuit(weights, x=x))
	print(circuit(weights, x=x))
	print(circuit(weights, x=x))
	print("==============================")