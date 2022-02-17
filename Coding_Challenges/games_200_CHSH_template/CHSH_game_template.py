#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np
import math


dev = qml.device("default.qubit", wires=2)


def prepare_entangled(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # QHACK #
    alpha = alpha / math.sqrt(alpha ** 2 + beta ** 2)
    qml.PauliY(2 * math.acos(alpha), wires=0)
    qml.CNOT(wires=[0, 1])
    # QHACK #


@qml.qnode(dev)
def chsh_circuit(theta_A0, theta_A1, theta_B0, theta_B1, x, y, alpha, beta):
    """Construct a circuit that implements Alice's and Bob's measurements in the rotated bases

    Args:
        - theta_A0 (float): angle that Alice chooses when she receives x=0
        - theta_A1 (float): angle that Alice chooses when she receives x=1
        - theta_B0 (float): angle that Bob chooses when he receives x=0
        - theta_B1 (float): angle that Bob chooses when he receives x=1
        - x (int): bit received by Alice
        - y (int): bit received by Bob
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (np.tensor): Probabilities of each basis state
    """

    prepare_entangled(alpha, beta)

    # QHACK #
    theta_A = [theta_A0, theta_A1]
    theta_B = [theta_B0, theta_B1]

    qml.RY(- 2 * theta_A[x], wires=0)
    qml.RY(- 2 * theta_B[y], wires=1)
    # QHACK #

    return qml.probs(wires=[0, 1])


def winning_prob(params, alpha, beta):
    """Define a function that returns the probability of Alice and Bob winning the game.

    Args:
        - params (list(float)): List containing [theta_A0,theta_A1,theta_B0,theta_B1]
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning the game
    """

    # QHACK #
    phi = params
    return (1/4
            * (3 * math.cos(phi) ** 2 + math.sin(3 * phi) ** 2
                + (2 * alpha * math.sqrt(1 - alpha ** 2) - 1) *
                (math.sin(4 * phi) * math.sin(2 * phi))
               ))

    # QHACK #


def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning
    """

    def cost(alpha, phi):
        """Define a cost function that only depends on params, given alpha and beta fixed"""
        return (-1/4
                * (3 * math.cos(phi) ** 2 + math.sin(3 * phi) ** 2
                   + (2 * alpha * math.sqrt(1 - alpha ** 2) - 1) *
                   (math.sin(4 * phi) * math.sin(2 * phi))
                   ))

    # QHACK #

    # Initialize parameters, choose an optimization method and number of steps
    alpha = alpha / math.sqrt(alpha ** 2 + beta ** 2)
    beta = math.sqrt(1 - alpha ** 2)
    phi = 0

    # QHACK #
    curr_cost = 0
    # set the initial parameter values
    while phi <= math.pi / 4:
        # update the circuit parameters
        # QHACK #
        if cost(alpha, phi) >= curr_cost:
            break
        curr_cost = cost(alpha, phi)
        phi += 1e-8
    assert phi <= math.pi / 4
    # QHACK #

    params = phi

    return winning_prob(params, alpha, beta)


if __name__ == '__main__':
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")
