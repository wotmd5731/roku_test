# -*- coding: utf-8 -*-
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os

import numpy as np
import copy 


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors -- output from policy function - a list of tuples of actions
            and their prior probability according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value, Q plus bonus u(P).
        Returns:
        A tuple of (action, next_node)
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        Arguments:
        leaf_value -- the value of subtree evaluation from the current player's perspective.        
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value) 

    def get_value(self, c_puct):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        """
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search.
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """Arguments:
        policy_value_fn -- a function that takes in a board state and outputs a list of (action, probability)
            tuples and also a score in [-1, 1] (i.e. the expected value of the end game score from 
            the current player's perspective) for the current player.
        c_puct -- a number in (0, inf) that controls how quickly exploration converges to the
            maximum-value policy, where a higher value means relying on the prior more
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.
        Arguments:
        state -- a copy of the state.
        """
        
        
    
        node = self._root
        while(1):            
            if node.is_leaf():
                break                
            # Greedily select next move.
            action, node = node.select(self._c_puct)  
            state.step(action)
#            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of (action, probability)
        # tuples p and also a score v in [-1, 1] for the current player.
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Runs all playouts sequentially and returns the available actions and their corresponding probabilities 
        Arguments:
        state -- the current state, including both game state and the current player.
        temp -- temperature parameter in (0, 1] that controls the level of exploration
        Returns:
        the available actions and the corresponding probabilities 
        """        
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
  
        # calc the move probabilities based on the visit counts at the root node
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))       
         
        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"
        
from model import PolicyValueNet

class Agent_MCTS(nn.Module):
    def __init__(self,args,c_puct,n_playout,self_play,shared_lr_mul,shared_g_cnt):
        super().__init__()
        self.cuda = args.cuda
        self._is_selfplay=self_play
        
        
        self.n_playout = n_playout # num of simulations for each move
        self.c_puct = c_puct
        
        self.play_batch_size = 1 
        self.check_freq = 50 
        self.game_batch_num = 1500
        # num of simulations used for the pure mcts, which is used as the opponent to evaluate the trained policy
        
        
        self.policy_value_net = PolicyValueNet(args.board_max,args.board_max,use_gpu=args.cuda)
        self.mcts = MCTS(self.policy_value_net.policy_value_fn, self.c_puct, self.n_playout)
        
        
#        self.main_dqn= DQN_model(args)
        
#        self.main_dqn.train()
#        self.target_dqn = DQN_rainbow(args)
#        self.target_dqn = target_model
#        self.target_dqn_update()
#        self.target_dqn.eval()
        
#        self.optimizer = optim.Adam(self.main_dqn.parameters(), lr=args.lr, eps=args.adam_eps)
        
      
    def reset_player(self):
        self.mcts.update_with_move(-1) 
        
    def save(self):
#        print('save')
        torch.save(self.policy_value_net.policy_value_net.state_dict(),'./net_param')
        return 1
    def model_update(self,model):
#        self.policy_value_net.policy_value_net.parameter_update(model)
        if self.cuda:
            self.policy_value_net.policy_value_net.load_state_dict(model.state_dict())
        else:
            self.policy_value_net.policy_value_net.load_state_dict(model.state_dict())
        
#    def save(self,path ='./param.p'):
#        torch.save(self.main_dqn.state_dict(),path)
#        
#    def load(self,path ='./param.p'):
#        if os.path.exists(path):
#            self.main_dqn.load_state_dict(torch.load(path))
#        else :
#            print("file not exist")
    
#    def target_dqn_update(self):
#        self.target_dqn.parameter_update(self.main_dqn)
    
   
    def get_action(self, board, temp=1e-3):
        sensible_moves = board.availables
        move_probs = np.zeros(board.width*board.height) # the pi vector returned by MCTS as in the alphaGo Zero paper
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs         
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for self-play training)
                if random.random() >0.3:
                    move = np.random.choice(sensible_moves)
                else:
#                move = np.random.choice(acts, p=0.7*probs + 0.3*np.random.dirichlet(0.3*np.ones(len(probs))))  
#                    move = np.random.choice(acts, p=probs)  
                    move = acts[np.argmax(probs)]
                    
                
                self.mcts.update_with_move(move) # update the root node and reuse the search tree
            else:
                # with the default temp=1e-3, thisZ is almost equivalent to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)       
                # reset the root node
                self.mcts.update_with_move(-1)             
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))
                
            return move, move_probs
        else:            
            print("WARNING: the board is full")

        
    
#    def train(self):
#        self.main_dqn.train()
#    def eval(self):
#        self.main_dqn.eval()
        
    
    
        
