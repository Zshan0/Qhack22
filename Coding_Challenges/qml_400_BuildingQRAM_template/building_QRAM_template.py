#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml
from numpy import cos, sin


def getMatrix(theta):
    """
    Returns the unitary matrix for rotation by given angle in Y axis.
    Args:
        - theta (float): angles to apply in the rotation.

    Returns:
        - (np.ndarray): Unitary Matrix.
    """
    return np.array([[cos(theta / 2), -sin(theta)], [sin(theta / 2), cos(theta)]])


def qRAM(thetas):
    """Function that generates the superposition state explained above given the thetas angles.

    Args:
        - thetas (list(float)): list of angles to apply in the rotations.

    Returns:
        - (list(complex)): final state.
    """

    # QHACK #

    # Use this space to create auxiliary functions if you need it.

    # QHACK #

    dev = qml.device("default.qubit", wires=range(4))

    @qml.qnode(dev)
    def circuit():

        # QHACK #

        # Create your circuit: the first three qubits will refer to the index, the fourth to the RY rotation.
        # Creating max superposition state for the first 3 qubits
        for i in range(3):
            qml.Hadamard(wires=i)

        for ind, theta in enumerate(thetas):
            bitString = "{0:b}".format(ind)
            bitString = bitString.rjust(3, '0')
            qml.ControlledQubitUnitary(
                getMatrix(theta), control_wires=range(3), wires=3, control_values=bitString
            )

        # QHACK #

        return qml.state()

    return circuit()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    thetas = np.array(inputs, dtype=float)

    output = qRAM(thetas)
    output = [float(i.real.round(6)) for i in output]
    print(*output, sep=",")
