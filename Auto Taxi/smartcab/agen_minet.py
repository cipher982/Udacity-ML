import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

# Defines initial parameters, not sure what to add here???    
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.possible_actions = [None, 'left', 'forward', 'right']
        self.qs = {}

        
        
# This is where the code for a new run resets itself
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required


# Here is where I implement the Q-Learning policy
    def optimal_action(self, state):
        all_qs = {action: self.qs.get((state, action), 0) for action in self.possible_actions}

        # First attempts at grabbing the best move, but couldn't pull values from dictionary
#        maxQ = all_qs[max(all_qs, key=all_qs.get)]
#        maxQ = max(all_qs.iterkeys(), key=lambda k: all_qs[k])
#        action = self.possible_actions[maxQ]

        optimal_actions = [action for action in self.possible_actions if all_qs[action] == max(all_qs.values())]
        return random.choice(optimal_actions) 
 
 # First attempt at creating a dictionary, but can not figure out how to pull values instead of keys
 

        
# This is where I can define the inputs and take the actions chosen above       
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

         # update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)

        # Pick optimal action and get reward
        action = self.optimal_action(self.state)
        reward = self.env.act(self, action)
        print reward
        print "The state is %s, and the reward is %d" % (self.qs.get(self.state, action), reward)
        # Finally update the Q-Values
        self.qs[(self.state, action)] = self.qs.get((self.state, action), 0) + reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]



# Program Starts here         
def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.01, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line



if __name__ == '__main__':
    run()
