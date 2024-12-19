# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            new_values = util.Counter()  # Initialize a new Counter to hold the updated values
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):  # Terminal states have a value of 0
                    new_values[state] = 0
                else:
                    max_q_value = float("-inf")
                    for action in self.mdp.getPossibleActions(state):
                        q_value = self.computeQValueFromValues(state, action)
                        if q_value > max_q_value:
                            max_q_value = q_value
                    new_values[state] = max_q_value
            self.values = new_values  # Update the values for the next iteration


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_value = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            discount_factor = self.discount
            q_value += prob * (reward + discount_factor * self.values[next_state])
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        best_action = None
        best_value = float("-inf")
        for action in self.mdp.getPossibleActions(state):
            q_value = self.computeQValueFromValues(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        num_states = len(states)
        for i in range(self.iterations):
            state_index = i % num_states  # Get the index of the state to update
            state = states[state_index]

            if self.mdp.isTerminal(state):
                continue  # Skip terminal states

            # Update the value of the current state based on the Bellman equation
            self.values[state] = self.computeQValueFromValues(state, self.computeActionFromValues(state))

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Step 1: Compute predecessors of all states
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = set()
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for successor, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                    predecessors[successor].add(state)

        # Step 2: Initialize an empty priority queue
        priority_queue = util.PriorityQueue()

        # Step 3: For each non-terminal state s
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                # Step 3a: Find the absolute difference between the current value and the highest Q-value
                diff = abs(self.values[state] - self.getMaxQValue(state))
                # Step 3b: Push state s into the priority queue with priority -diff
                priority_queue.update(state, -diff)

        # Step 4: For each iteration
        for iteration in range(self.iterations):
            # Step 4a: If priority queue is empty, terminate
            if priority_queue.isEmpty():
                break
            # Step 4b: Pop a state s off the priority queue
            state = priority_queue.pop()
            # Step 4c: Update the value of state s (if it is not a terminal state)
            if not self.mdp.isTerminal(state):
                self.values[state] = self.getMaxQValue(state)
            # Step 4d: For each predecessor p of state s
            for predecessor in predecessors[state]:
                # Step 4d-i: Find the absolute difference between current value of p and highest Q-value
                diff = abs(self.values[predecessor] - self.getMaxQValue(predecessor))
                # Step 4d-ii: If diff > theta, push p into the priority queue with priority -diff
                if diff > self.theta:
                    priority_queue.update(predecessor, -diff)

    def getMaxQValue(self, state):
            """
            Compute the maximum Q-value across all possible actions from a given state.

            :param state: The state for which to compute the maximum Q-value
            :return: The maximum Q-value
            """
            if self.mdp.isTerminal(state):
                return 0
            return max(self.getQValue(state, action) for action in self.mdp.getPossibleActions(state))

    def getQValue(self, state, action):
            """
            Compute the Q-value of a state-action pair.

            :param state: The state
            :param action: The action
            :return: The Q-value of the state-action pair
            """
            q_value = 0
            for successor, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                reward = self.mdp.getReward(state, action, successor)
                q_value += probability * (reward + self.discount * self.values[successor])
            return q_value
