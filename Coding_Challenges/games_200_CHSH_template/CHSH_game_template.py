#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np
import math


dev = qml.device("default.qubit", wires=2)


def normalize(alpha, beta):
    norm = math.sqrt(alpha**2 + beta**2)
    return alpha / norm, beta / norm


def prepare_entangled(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # QHACK #
    alpha, beta = normalize(alpha, beta)
    qml.RY(2 * math.acos(alpha), wires=0)
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
    alpha, beta = normalize(alpha, beta)
    prepare_entangled(alpha, beta)

    # QHACK #
    theta_A = [theta_A0, theta_A1]
    theta_B = [theta_B0, theta_B1]

    qml.RY(-2 * theta_A[x], wires=0)
    qml.RY(-2 * theta_B[y], wires=1)
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
    alpha, beta = normalize(alpha, beta)

    # QHACK #
    def probs_theta(theta_A, theta_B):
        term1 = math.cos(theta_B - theta_A) ** 2
        term21 = (4 * alpha * beta) - 2
        term22 = (
            math.cos(theta_A)
            * math.sin(theta_A)
            * math.sin(theta_B)
            * math.cos(theta_B)
        )
        return term1 + (term21 * term22)

    val = probs_theta(params[0], params[2])
    val += probs_theta(params[0], params[3])
    val += probs_theta(params[1], params[2])
    val += 1 - probs_theta(params[1], params[3])

    return val / 4

    # QHACK #


def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning
    """
    alpha, beta = normalize(alpha, beta)

    def cost(params):
        """Define a cost function that only depends on params, given alpha and beta fixed"""
        return -winning_prob(params, alpha, beta)

    def get_params(phi):
        return [0, 2 * phi, phi, -phi]

    # QHACK #

    # Initialize parameters, choose an optimization method and number of steps
    phi = 0
    precision = 1e-6

    # QHACK #
    curr_cost = 0
    # set the initial parameter values
    while phi <= math.pi / 4:
        # update the circuit parameters
        # QHACK #
        params = get_params(phi)
        if cost(params) >= curr_cost:
            break
        curr_cost = cost(params)
        phi += precision

    assert phi <= math.pi / 4
    # QHACK #

    return winning_prob(get_params(phi), alpha, beta)


if __name__ == "__main__":
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")
