#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=1, shots=1)


@qml.qnode(dev)
def is_bomb(angle):
    """Construct a circuit at implements a one shot measurement at the bomb.

    Args:
        - angle (float): transmissivity of the Beam splitter, corresponding
        to a rotation around the Y axis.

    Returns:
        - (np.ndarray): a length-1 array representing result of the one-shot measurement
    """

    # QHACK #
    splitter(angle)
    # QHACK #

    return qml.sample(qml.PauliZ(0))


def splitter(angle):
    qml.RY(2 * angle, wires=0)

@qml.qnode(dev)
def bomb_tester(angle):
    """Construct a circuit that implements a final one-shot measurement, given that the bomb does not explode

    Args:
        - angle (float): transmissivity of the Beam splitter right before the final detectors

    Returns:
        - (np.ndarray): a length-1 array representing result of the one-shot measurement
    """

    # QHACK #
    splitter(angle)

    # QHACK #

    return qml.sample(qml.PauliZ(0))


def simulate(angle, n):
    """Concatenate n bomb circuits and a final measurement, and return the results of 10000 one-shot measurements

    Args:
        - angle (float): transmissivity of all the beam splitters, taken to be identical.
        - n (int): number of bomb circuits concatenated

    Returns:
        - (float): number of bombs successfully tested / number of bombs that didn't explode.
    """

    # QHACK #
    explosion, non_explosion_detection, no_explosion = 0, 0, 0
    shots = 10000

    for _ in range(shots):
        did_explode = False
        for _ in range(n):
            bomb = is_bomb(angle)
            if bomb == 1:
                # the bomb exploded
                did_explode = True
                break

        if not did_explode:
            test = bomb_tester(angle)
            no_explosion += 1
            if test == -1:
                # bomb detected without explosion
                non_explosion_detection += 1
        else:
            did_explode += 1


    return non_explosion_detection / no_explosion

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = simulate(float(inputs[0]), int(inputs[1]))
    print(f"{output}")
