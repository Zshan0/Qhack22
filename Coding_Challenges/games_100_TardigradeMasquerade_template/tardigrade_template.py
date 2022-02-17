import sys
import pennylane as qml
from pennylane import numpy as np


def second_renyi_entropy(rho):
    """Computes the second Renyi entropy of a given density matrix."""
    # DO NOT MODIFY anything in this code block
    rho_diag_2 = np.diagonal(rho) ** 2.0
    return -np.real(np.log(np.sum(rho_diag_2)))


def compute_entanglement(theta):
    """Computes the second Renyi entropy of circuits with and without a tardigrade present.

    Args:
        - theta (float): the angle that defines the state psi_ABT

    Returns:
        - (float): The entanglement entropy of qubit B with no tardigrade
        initially present
        - (float): The entanglement entropy of qubit B where the tardigrade
        was initially present
    """

    dev = qml.device("default.qubit", wires=3)

    # QHACK #

    @qml.qnode(dev)
    def normal_circuit(theta):

        qml.PauliX(1)
        qml.Hadamard(0)
        qml.CNOT(wires=[0, 1])
        # max ent state prepped

        return qml.density_matrix(1)

    # QHACK #

    @qml.qnode(dev)
    def tardigrade_circuit(theta):

        CRy = qml.ctrl(qml.RY, control=[0])
        CX = qml.ctrl(qml.PauliX, control=[0])
        CCX = qml.ctrl(qml.PauliX, control=[0, 1])

        qml.Hadamard(0)
        CRy(theta, wires=[1])
        CX(wires=[1])
        CX(wires=[2])
        CCX(wires=[2])
        qml.PauliX(0)

        return qml.density_matrix(1)

    return second_renyi_entropy(normal_circuit(theta)), second_renyi_entropy(
        tardigrade_circuit(theta)
    )


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    theta = np.array(sys.stdin.read(), dtype=float)

    S2_without_tardigrade, S2_with_tardigrade = compute_entanglement(theta)
    print(*[S2_without_tardigrade, S2_with_tardigrade], sep=",")
