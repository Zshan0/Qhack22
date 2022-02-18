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
    qml.RY(2 * math.acos(alpha), wires=0)
    qml.CNOT(wires=[0, 1])
    # QHACK #


# @qml.qnode(dev)
@qml.qnode(dev, interface='autograd')
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
    # QHACK #
    answer = 0

    probs = chsh_circuit(*params, x=0, y=0, alpha=alpha, beta=beta)
    answer += (probs[0] + probs[3]) # 0.0 = 0 + 0  or 1 + 1
    probs = chsh_circuit(*params, x=0, y=1, alpha=alpha, beta=beta)
    answer += (probs[0] + probs[3]) # 0.1 = 0 + 0  or 1 + 1
    probs = chsh_circuit(*params, x=1, y=0, alpha=alpha, beta=beta)
    answer += (probs[0] + probs[3]) # 1.0 = 0 + 0  or 1 + 1
    probs = chsh_circuit(*params, x=1, y=1, alpha=alpha, beta=beta)
    answer += (probs[1] + probs[2]) # 1.1 = 1 + 0  or 1 + 0
    
    answer = answer / 4
    return answer
    

    # QHACK #


def cost_new(params, alpha, beta):
    """Define a function that returns the probability of Alice and Bob winning the game.
    Args:
        - params (list(float)): List containing [theta_A0,theta_A1,theta_B0,theta_B1]
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    Returns:
        - (float): Probability of winning the game
    """
    params = params._value

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


def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning
    """
    def cost(params):
        """Define a cost function that only depends on params, given alpha and beta fixed"""
        return -cost_new(params, alpha, beta)


    # QHACK #

    # Initialize parameters, choose an optimization method and number of steps
    alpha, beta = normalize(alpha, beta)
    init_params = np.random.rand(4, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=1e-4)
    steps = int(5000)

    # QHACK #
    
    # set the initial parameter values
    params = init_params

    for _ in range(steps):
        # update the circuit parameters 
        # QHACK #

        params = opt.step(cost, params)

        # QHACK #

    return winning_prob(params, alpha, beta)


if __name__ == "__main__":
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")
