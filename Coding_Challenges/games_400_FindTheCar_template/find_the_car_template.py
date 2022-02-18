#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


dev = qml.device("default.qubit", wires=[0, 1, "sol"], shots=1)


def find_the_car(oracle):
    """Function which, given an oracle, returns which door that the car is behind.

    Args:
        - oracle (function): function that will act as an oracle. The first two qubits (0,1)
        will refer to the door and the third ("sol") to the answer.

    Returns:
        - (int): 0, 1, 2, or 3. The door that the car is behind.
    """

    @qml.qnode(dev)
    def circuit1():
        # QHACK #

        qml.Hadamard(1)
        qml.PauliX("sol")
        qml.Hadamard("sol")

        oracle()

        qml.Hadamard(1)

        # QHACK #
        return qml.sample(wires=1)

    @qml.qnode(dev)
    def circuit2(pos):
        # QHACK #

        if pos == 0:
            qml.PauliX(0)
        oracle()

        # QHACK #
        return qml.sample(wires="sol")

    sol1 = circuit1()
    sol2 = circuit2(sol1)

    answer = 0
    if sol1 == 0:
        answer += 2
    if sol2 == 0:
        answer += 1

    return answer

    # QHACK #

    # process sol1 and sol2 to determine which door the car is behind.

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    def oracle():
        if numbers[0] == 1:
            qml.PauliX(wires=0)
        if numbers[1] == 1:
            qml.PauliX(wires=1)
        qml.Toffoli(wires=[0, 1, "sol"])
        if numbers[0] == 1:
            qml.PauliX(wires=0)
        if numbers[1] == 1:
            qml.PauliX(wires=1)

    output = find_the_car(oracle)
    print(f"{output}")
