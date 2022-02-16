#! /usr/bin/python3

import sys
import math
import pennylane as qml
from pennylane import numpy as np


def error_wire(circuit_output):
    """Function that returns an error readout.

    Args:
        - circuit_output (?): the output of the `circuit` function.

    Returns:
        - (np.ndarray): a length-4 array that reveals the statistics of the
        error channel. It should display your algorithm's statistical prediction for
        whether an error occurred on wire `k` (k in {1,2,3}). The zeroth element represents
        the probability that a bitflip error does not occur.

        e.g., [0.28, 0.0, 0.72, 0.0] means a 28% chance no bitflip error occurs, but if one
        does occur it occurs on qubit #2 with a 72% chance.
    """

    # QHACK #
    # process the circuit output here and return which qubit was the victim of a bitflip error!
    p = -1
    alpha = -1
    wire = -1
    out = circuit_output

    # finding p
    if out[0] == 0 and out[7] == 0:
        p = 1
    elif out[0] == 0 or out[7] == 0:
        alpha = 0
        p = 1 - max(out[0], out[7])
    else:
        # out[0] or out[7] are not 0
        k = out[0]/out[7]
        alpha = math.sqrt(k / (1 + k))
        p = 1 - out[0]/(alpha ** 2)

    if p == 0:
        return [1, 0, 0, 0]

    # finding wire which is tampered
    # p != 0
    for i in range(0, 3):
        if out[3 - i] != 0 or out[3 + i + 1] != 0:
            wire = i
            break
    # print(circuit_output)
    # print(p, alpha, wire)
    assert wire != -1 and p != -1
    ans = [1 - p, 0, 0, 0]
    ans[wire + 1] = p
    return ans
    # QHACK #


dev = qml.device("default.mixed", wires=3)


@qml.qnode(dev)
def circuit(p, alpha, tampered_wire):
    """A quantum circuit that will be able to identify bitflip errors.

    DO NOT MODIFY any already-written lines in this function.

    Args:
        p (float): The bit flip probability
        alpha (float): The parameter used to calculate `density_matrix(alpha)`
        tampered_wire (int): The wire that may or may not be flipped (zero-index)

    Returns:
        Some expectation value, state, probs, ... you decide!
    """

    qml.QubitDensityMatrix(density_matrix(alpha), wires=[0, 1, 2])

    # QHACK #

    # put any input processing gates here
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])
    qml.BitFlip(p, wires=int(tampered_wire))

    # put any gates here after the bitflip error has occurred
    # qml.CNOT(wires=[0, 1])
    # qml.CNOT(wires=[0, 2])
    # qml.Toffoli(wires=[1, 2, 0])

    return qml.probs(wires=[0, 1, 2])
    # QHACK #e


def density_matrix(alpha):
    """Creates a density matrix from a pure state."""
    # DO NOT MODIFY anything in this code block
    psi = alpha * np.array([1, 0], dtype=float) + np.sqrt(1 - alpha**2) * np.array(
        [0, 1], dtype=float
    )
    psi = np.kron(psi, np.array([1, 0, 0, 0], dtype=float))
    return np.outer(psi, np.conj(psi))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = np.array(sys.stdin.read().split(","), dtype=float)
    p, alpha, tampered_wire = inputs[0], inputs[1], int(inputs[2])

    error_readout = np.zeros(4, dtype=float)
    circuit_output = circuit(p, alpha, tampered_wire)
    error_readout = error_wire(circuit_output)

    print(*error_readout, sep=",")
