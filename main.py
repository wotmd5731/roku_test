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
from multiprocessing import Value , Queue, Lock
import datetime
#import torchvision.transforms as T
from collections import defaultdict, deque
import sys
import os 
import pickle

import argparse
"""
X - MCTS
"""
#"""
#define test function
#"""
#from plot import _plot_line
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
    
    

def human_process(args,share_model,rank,self_play,shared_lr_mul,shared_g_cnt,shared_q,lock):
    print('human play')
    self_play=False
    board_max = args.board_max
    from agent import Agent_MCTS
    agent = Agent_MCTS(args,5,800,self_play,shared_lr_mul,shared_g_cnt)
    with lock:
        agent.model_update(share_model)
            
    from checkerboard import Checkerboard, BoardRender
    board = Checkerboard(board_max,args.n_rows)
    board_render = BoardRender(board_max,render_off=False,inline_draw=True)
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
            move, move_probs = agent.get_action(board)
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
    

def act_process(args,share_model,rank,self_play,shared_lr_mul,shared_g_cnt,shared_q,lock):
    print(rank)
    board_max = args.board_max
    
    from agent import Agent_MCTS
    agent = Agent_MCTS(args,5,100,self_play,shared_lr_mul,shared_g_cnt)
    from checkerboard import Checkerboard, BoardRender
    board = Checkerboard(board_max,args.n_rows)
    board_render = BoardRender(board_max,render_off=True,inline_draw=False)
    board_render.clear()
    
    Ts =[]
    Tloss =[]
    Tentropy =[]
    try:
        for episode in range(10000):
            start_time = time.time()
            
            with lock:
                agent.model_update(share_model)
            
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
                move, move_probs = agent.get_action(board, temp=1.0)
                # store the data
                states.append(board.current_state())
                mcts_probs.append(move_probs)
                current_players.append(board.current_player)
                # perform a move
                board.step(move)
                board_render.draw(board.states)
                end, winner = board.game_end()
                if end:
#                    time.sleep(1)
                    # winner from the perspective of the current player of each state
                    winners_z = np.zeros(len(current_players))  
                    if winner != -1:
                        winners_z[np.array(current_players) == winner] = 1.0
                        winners_z[np.array(current_players) != winner] = -1.0
                    #reset MCTS root node
                    agent.reset_player() 
                    if winner != -1:
                        print(rank, "Game end. Winner is player:", winner, 'total_step :',step, 'time:',time.time()-start_time)
                    else:
                        print(rank, "Game end. Tie", 'total_step :',step,'time:',time.time()-start_time)
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
        print(rank,'except end')



def learn_process(args,share_model,shared_lr_mul,shared_g_cnt,shared_q,lock):
    from model import PolicyValueNet
    epochs = 1
    kl_targ = 0.025
    lr_multiplier = shared_lr_mul  # adaptively adjust the learning rate based on KL
    g_cnt = shared_g_cnt
    learn_rate = args.lr
    batch_size = args.batch_size # mini-batch size for training
    policy_value_net = PolicyValueNet(args.board_max,args.board_max,use_gpu=args.cuda)
    policy_value_net.policy_value_net.load_state_dict(share_model.state_dict())
    
    data_buffer = deque(maxlen=args.memory_capacity)
    
    try:
        with open('qqq.dat','rb') as qq:
            temp_buffer = pickle.load(qq)
            print('load buffer length: ',len(temp_buffer))
            data_buffer.extend(temp_buffer)
    except:
        pass
    
    
    def learn(policy_value_net,rank,data_buffer):
        """update the policy-value net"""
        mini_batch = random.sample(data_buffer, batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]            
        old_probs, old_v = policy_value_net.policy_value(state_batch) 
        for i in range(epochs): 
            
            loss, entropy = policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, learn_rate*lr_multiplier.value)
            new_probs, new_v = policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))  
            if kl > kl_targ * 4:   # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > kl_targ * 2 and lr_multiplier.value > 0.1:
            lr_multiplier.value /= 1.5
        elif kl < kl_targ / 2 and lr_multiplier.value < 10:
            lr_multiplier.value *= 1.5
#        explained_var_old =  1 - np.var(np.array(winner_batch) - old_v.flatten())/np.var(np.array(winner_batch))
#        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten())/np.var(np.array(winner_batch))     
#        ss = "rank:{} c:{} kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{},explained_var_old:{:.3f},explained_var_new:{:.3f}  {} ".format(
#                rank,self.g_cnt.value,kl, self.lr_multiplier.value, loss, entropy, explained_var_old, explained_var_new,datetime.datetime.now())
        ss = "r:{} c:{} kl:{:.5f},lr_mul:{:.3f},loss:{},ent:{}   {} ".format(
                rank,g_cnt.value,kl, lr_multiplier.value, loss, entropy, datetime.datetime.now())
        
        g_cnt.value +=1
        if g_cnt.value%100 == 0:
            with open('log.txt','a') as f:
                f.write(ss+'\n')
        print('\r'+ ss,end='',flush=True)
#        print(ss)
        return loss, entropy
    
    print('learner')

    
            
    try:
        while True:      
            if not shared_q.empty():
                print('extend',len(data_buffer) ,'>',args.learn_start)
                data_buffer.extend(shared_q.get())
            
                if len(data_buffer) > args.learn_start:
                    for i in range(7):
                        loss, entropy = learn(policy_value_net,0,data_buffer)
                    with lock:
                        share_model.load_state_dict(policy_value_net.policy_value_net.state_dict())
            else:
                print('buffer fill :',len(data_buffer) ,'>',args.learn_start, end='\r')
                
            if shared_g_cnt.value%1000 ==0:
                print('leanr_save')
                torch.save(policy_value_net.policy_value_net.state_dict(),'./net_param')
            time.sleep(1)
    #                    list_loss.append(loss)
    #                    list_entropy.append(entropy)
    #                print('loss : ',loss,' entropy : ',entropy)
    except:
        torch.save(policy_value_net.policy_value_net.state_dict(),'./net_param')
        with open('qqq.dat','wb') as qq:
            pickle.dump(data_buffer,qq,-1)
            print('pickle dump')
        print(999,'except save')

if __name__ == '__main__':
    
    board_max = 7
    n_rows = 4
    
    
    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--name', type=str, default='main_rainbow_multi.p', help='stored name')
#    parser.add_argument('--epsilon', type=float, default=0.05, help='random action select probability')
    #parser.add_argument('--render', type=bool, default=True, help='enable rendering')
#    parser.add_argument('--render', type=bool, default=False, help='enable rendering')
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
    parser.add_argument('--memory-capacity', type=int, default=600000, metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--learn-start', type=int, default=500000 , metavar='STEPS', help='Number of steps before starting training')
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
    args.disable_cuda = True
        
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
    lock = Lock()
    
#    share_model.cuda()
    shared_lr_mul = Value('d',1)
    shared_g_cnt = Value('i',1)
    shared_q = Queue(maxsize=10000)
    
    try:
        share_model.load_state_dict(torch.load('./net_param'))
        print('load')
    except:
        print('load fail')
     
    processes = []
    
    human_process(args,share_model,0,self_play,shared_lr_mul,shared_g_cnt,shared_q,lock)
    
#    num_processes = 10
#    for rank in range(num_processes):
#        p = mp.Process(target=act_process, args=(args,share_model,rank,self_play,shared_lr_mul,shared_g_cnt,shared_q,lock))
#        p.start()
#        processes.append(p)
#    try:
##        act_process(args,share_model,0,self_play,shared_lr_mul,shared_g_cnt,shared_q,lock)
#        learn_process(args,share_model,shared_lr_mul,shared_g_cnt,shared_q,lock)
#    except:
#        pass
##        shared_q.close()
##        shared_q.join_thread()
#    for ps in processes:
#        ps.terminate()
#        ps.join()	
#            
#        
    
    
