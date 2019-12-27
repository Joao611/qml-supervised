import pennylane as qml
from pennylane import numpy as np

num_qubits = 5
shots = 10
dev = qml.device("default.qubit", wires=num_qubits, shots=shots)

def state_preparation():
	qml.BasisState(np.zeros(num_qubits), wires=[i for i in range(num_qubits)])

# def circU():
	

def feature_map():
	for i in range(num_qubits):
		qml.Hadamard(i)
	# circU()
	for i in range(num_qubits):
		qml.Hadamard(i)
	# circU()

# def weights_variational():
	

@qml.qnode(dev)
def circuit():
	state_preparation()
	feature_map()
	# weights_variational()
	return [qml.sample(qml.PauliZ(wire)) for wire in range(num_qubits)]

print(circuit())