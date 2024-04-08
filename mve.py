import jax
from jax import numpy as np
import pennylane as qml
import optax

jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)

h2_dataset = qml.data.load("qchem", molname="H2", bondlength=0.742, basis="STO-3G")
h2 = h2_dataset[0]
H, qubits = h2.hamiltonian, len(h2.hamiltonian.wires)


dev = qml.device("lightning.qubit", wires=qubits)

hf = h2.hf_state

@qml.qnode(dev)
def circuit(param):
    qml.BasisState(hf, wires=range(qubits))
    qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
    return qml.expval(H)
@qml.qnode(dev)
def circuit_2(state, param):
    qml.BasisState(state, wires=range(qubits))
    qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
    return qml.expval(H)

max_iterations = 100
conv_tol = 1e-06
opt = optax.sgd(learning_rate=0.4)
theta = np.array(0.)
angle = [theta]
opt_state = opt.init(theta)

for n in range(max_iterations):
    # gradient = jax.grad(circuit)(theta)
    gradient = jax.grad(circuit_2)(h2.hf_state, theta)
    updates, opt_state = opt.update(gradient, opt_state)
    theta = optax.apply_updates(theta, updates)

