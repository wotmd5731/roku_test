# -*- coding: utf-8 -*-
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
from multiprocessing import Value , Queue

#import torchvision.transforms as T
from collections import defaultdict, deque
import sys
import os 

import argparse
"""
X - MCTS
"""
#"""
#define test function
#"""
from plot import _plot_line
def get_equi_data(play_data,board_height,board_width):
    """
    augment the data set by rotation and flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]"""
#        [state, mcts_porb, winner ]  = zip(*play_data)
    extend_data = []
    for state, mcts_porb, winner in play_data:
        for i in [1,2,3,4]:
            # rotate counterclockwise 
            equi_state = np.array([np.rot90(s,i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(board_height, board_width)), i)
            extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
    return extend_data
    
    

def human_process(args,share_model,rank,self_play,shared_lr_mul,shared_g_cnt,shared_q):
    print('human play')
    board_max = args.board_max
    
    from agent import Agent_MCTS
    agent = Agent_MCTS(args,share_model,self_play,shared_lr_mul,shared_g_cnt)
    from checkerboard import Checkerboard, BoardRender
    board = Checkerboard(board_max,args.n_rows)
    board_render = BoardRender(board_max,render_off=False,inline_draw=False)
    board.reset()
    board_render.clear()
    board_render.draw(board.states)
    
    p1, p2 = board.players
    player = input('select player 1: balck , 2 : white')
    if player=='1':
        play_step = 1
    else:
        play_step = 0
    for step in range(10000):
        if step//2 %2 == play_step:
            ss =input('input x,y:')
            pos = ss.split(',')
            if pos == 'q' :
                return
            move = int(pos[0])+int(pos[1])*board_max
            print('movd ',move)
        else:            
            move, move_probs = agent.get_action(board, return_prob=1)
        board.step(move)
        board_render.draw(board.states)
        end, winner = board.game_end()
        if end:
            # winner from the perspective of the current player of each state
            agent.reset_player() 
            if winner != -1:
                print("Game end. Winner is player:", winner)
            else:
                print("Game end. Tie")
#                return winner, zip(states, mcts_probs, winners_z)
            return
    

def act_process(args,share_model,rank,self_play,shared_lr_mul,shared_g_cnt,shared_q):
    print(rank)
    board_max = args.board_max
    
    from agent import Agent_MCTS
    agent = Agent_MCTS(args,share_model,self_play,shared_lr_mul,shared_g_cnt)
    from checkerboard import Checkerboard, BoardRender
    board = Checkerboard(board_max,args.n_rows)
    board_render = BoardRender(board_max,render_off=True,inline_draw=False)
    board_render.clear()
    
    
    
    
    

    Ts =[]
    Tloss =[]
    Tentropy =[]
    try:
        for episode in range(10000):
            random.seed(time.time())
            board.reset()
            board_render.clear()
            board_render.draw(board.states)
            
            """ start a self-play game using a MCTS player, reuse the search tree
            store the self-play data: (state, mcts_probs, z)
            """
            p1, p2 = board.players
            states, mcts_probs, current_players = [], [], []    
#            list_loss = []
#            list_entropy = []
            for step in range(10000):
                move, move_probs = agent.get_action(board, temp=1.0, return_prob=1)
                # store the data
                states.append(board.current_state())
                mcts_probs.append(move_probs)
                current_players.append(board.current_player)
                # perform a move
                board.step(move)
                board_render.draw(board.states)
                end, winner = board.game_end()
                if end:
                    # winner from the perspective of the current player of each state
                    winners_z = np.zeros(len(current_players))  
                    if winner != -1:
                        winners_z[np.array(current_players) == winner] = 1.0
                        winners_z[np.array(current_players) != winner] = -1.0
                    #reset MCTS root node
                    agent.reset_player() 
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
    #                return winner, zip(states, mcts_probs, winners_z)
                    play_data = zip(states, mcts_probs, winners_z)
                    ex_play_data = get_equi_data(play_data,board_max,board_max)
                    shared_q.put(ex_play_data)
                    break
                
#            # plot_data
#            if len(data_buffer) > args.batch_size and len(list_loss)!=0:
#                
#                Ts.append(episode)
#                Tloss.append(list_loss)
#                Tentropy.append(list_entropy)
#                _plot_line(Ts, Tloss, 'loss', path='./')
#                _plot_line(Ts, Tentropy, 'entropy', path='./')
            
            episode += 1
    except:
        print('except end')
        


    def learn_process(args,share_model,shared_lr_mul,shared_g_cnt,shared_q):
        data_buffer = deque(maxlen=args.memory_capacity)
        
        if not shared_q.empty():
            data_buffer.extend(shared_q.get())
        
        if len(data_buffer) > args.batch_size:
            loss, entropy = agent.learn(rank,data_buffer)
#                    list_loss.append(loss)
#                    list_entropy.append(entropy)
#                print('loss : ',loss,' entropy : ',entropy)

        if rank==0 :agent.save()


if __name__ == '__main__':
    
    board_max = 7
    n_rows = 4
    
    
    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--name', type=str, default='main_rainbow_multi.p', help='stored name')
    parser.add_argument('--epsilon', type=float, default=0.05, help='random action select probability')
    #parser.add_argument('--render', type=bool, default=True, help='enable rendering')
    parser.add_argument('--render', type=bool, default=False, help='enable rendering')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    #parser.add_argument('--game', type=str, default='CartPole-v1', help='gym game')
    #parser.add_argument('--game', type=str, default='Acrobot-v1', help='gym game')
    #parser.add_argument('--game', type=str, default='MountainCar-v0', help='gym game')
    parser.add_argument('--max-step', type=int, default=board_max*board_max, metavar='STEPS', help='Number of training steps (4x number of frames)')
    parser.add_argument('--action-space', type=int, default=board_max*board_max ,help='game action space')
    parser.add_argument('--n-rows', type=int, default=n_rows ,help='game rows')
    parser.add_argument('--board-max', type=int, default=board_max ,help='game board')
    parser.add_argument('--state-space', type=int, default=board_max*board_max ,help='game action space')
    parser.add_argument('--max-episode-length', type=int, default=100000, metavar='LENGTH', help='Max episode length (0 to disable)')
#    parser.add_argument('--history-length', type=int, default=1, metavar='T', help='Number of consecutive states processed')
#    parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
#    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
#    parser.add_argument('--atoms', type=int, default=11, metavar='C', help='Discretised size of value distribution')
#    parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
#    parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
    #parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--memory-capacity', type=int, default=500000, metavar='CAPACITY', help='Experience replay memory capacity')
#    parser.add_argument('--learn-start', type=int, default=1 , metavar='STEPS', help='Number of steps before starting training')
#    parser.add_argument('--replay-interval', type=int, default=1, metavar='k', help='Frequency of sampling from memory')
#    parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent')
#    parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
    #parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
#    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
#    parser.add_argument('--target-update-interval', type=int, default=1, metavar='τ', help='Number of steps after which to update target network')
#    parser.add_argument('--reward-clip', type=int, default=10, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
#    parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
#    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=256, metavar='SIZE', help='Batch size')
#    parser.add_argument('--max-gradient-norm', type=float, default=10, metavar='VALUE', help='Max value of gradient L2 norm for gradient clipping')
#    parser.add_argument('--save-interval', type=int, default=1000, metavar='SAVE', help='Save interval')
    #parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
#    parser.add_argument('--evaluation-interval', type=int, default=20, metavar='STEPS', help='Number of training steps between evaluations')
    #parser.add_argument('--evaluation-episodes', type=int, default=1, metavar='N', help='Number of evaluation episodes to average over')
    #parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
    #parser.add_argument('--log-interval', type=int, default=25000, metavar='STEPS', help='Number of training steps between logging status')
    
    # Setup
    args = parser.parse_args()
    " disable cuda "
    args.disable_cuda = False
        
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
      print(' ' * 26 + k + ': ' + str(v))
    args.cuda = torch.cuda.is_available() and not args.disable_cuda
    torch.manual_seed(random.randint(1, 10000))
    if args.cuda:
      torch.cuda.manual_seed(random.randint(1, 10000))
    print('cuda : ',args.cuda)
    
    
    
    self_play = True
    
    from model import Net
    
    share_model = Net(args.board_max, args.board_max)
    share_model.share_memory()
    shared_lr_mul = Value('d',1)
    shared_g_cnt = Value('i',0)
    shared_q = Queue(maxsize=10000)
    
    if True:
        try:
            share_model.load_state_dict(torch.load('./net_param'))
            print('load')
        except:
            print('load fail')
            pass    
    act_process(args,share_model,0,self_play,shared_lr_mul,shared_g_cnt,shared_q)

#    num_processes = 7
#    processes = []
#    for rank in range(num_processes):
#        p = mp.Process(target=act_process, args=(args,share_model,board_max,n_rows,rank,self_play,shared_lr_mul,shared_g_cnt))
#        p.start()
#        processes.append(p)
#    for p in processes:
#        p.join()