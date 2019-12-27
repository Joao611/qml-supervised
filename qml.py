import pennylane as qml
from pennylane import numpy as np

# Circuit is hardcoded in several places to only have 2 qubits, despite the following variables.
# For instance, the feature_map() only takes in 1 value "phi",
# whereas multiple values would be needed for more qubits.
num_qubits = 2
shots = 10

dev = qml.device("default.qubit", wires=num_qubits, shots=shots)

def state_preparation():
	qml.BasisState(np.zeros(num_qubits), wires=[i for i in range(num_qubits)])

def circU(phi=None):
	qml.CNOT(wires=[0, 1])
	qml.RZ(phi, wires=1)
	qml.CNOT(wires=[0, 1])

def feature_map(phi=None):
	for i in range(num_qubits):
		qml.Hadamard(i)
	circU(phi)
	for i in range(num_qubits):
		qml.Hadamard(i)
	circU(phi)

# def weights_variational():
	

@qml.qnode(dev)
def circuit(phi=None):
	state_preparation()
	feature_map(phi)
	# weights_variational()
	return [qml.sample(qml.PauliZ(wire)) for wire in range(num_qubits)]

print(circuit(phi=0))