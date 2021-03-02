import random

def Q_learning(state_map, population_map, action, reward, init_point, learning_rate = 0.3, gamma = 0.9, episode = 1000, init_epsilon = 0.9):
    Q_table = [[[0 for _ in range(len(action))] for _ in range(len(state_map[0]))] for _ in range(len(state_map))]
    # number of action per cell

    # learning
    for i in range(episode):
        present_state = init_point
        epsilon = init_epsilon/(i+1)
        episode_done = False

        while True:
            action_index = epsilon_greedy_action_selection(Q_table, present_state, action, epsilon)
            next_state = list(map(lambda x,y : x+y, present_state, action[action_index]))
            try: 
                if next_state[0] < 0 or next_state[1] < 0:
                    # outside of index of array map
                    present_reward = -9999
                    episode_done = True
                else:
                    # inside of map
                    present_reward = reward[state_map[next_state[0]][next_state[1]]]
                    if callable(present_reward):
                        present_reward = present_reward(population_map[next_state[0]][next_state[1]])
                    episode_done = state_map[next_state[0]][next_state[1]] == 1 or \
                        state_map[next_state[0]][next_state[1]] == 2
            except:
                # outside of index of array map
                present_reward = -9999
                episode_done = True
                
            if episode_done :
                Q_table[present_state[0]][present_state[1]][action_index] = \
                    (1-learning_rate) * Q_table[present_state[0]][present_state[1]][action_index] + \
                        learning_rate * (present_reward)
                break # one episode done
            else:
                maximum_Q_value = Q_table[next_state[0]][next_state[1]][epsilon_greedy_action_selection(Q_table, next_state, action, 0)]

                Q_table[present_state[0]][present_state[1]][action_index] = \
                    (1-learning_rate) * Q_table[present_state[0]][present_state[1]][action_index] + \
                        learning_rate * ( present_reward + gamma * maximum_Q_value )
                present_state = next_state

    # find best exit
    best_exit = []
    path_to_exit = []
    
    present_state = init_point
    move = 0

    while move<50:
        path_to_exit.append(present_state)

        if state_map[present_state[0]][present_state[1]] == 1:
            best_exit = present_state
            break
        
        action_index = epsilon_greedy_action_selection(Q_table, present_state, action, 0)
        present_state = list(map(lambda x,y : x+y, present_state, action[action_index]))

        move += 1

    return best_exit,path_to_exit

def epsilon_greedy_action_selection(Q_table, state, action, epsilon):
    # return action index 
    if epsilon > random.random():
        return random.randrange(len(action))
    else:
        return max(enumerate(Q_table[state[0]][state[1]]), key = lambda x : x[1])[0]

def path_reward(n):
    return -max(min(5,n-5), 0.01)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    state_map = [ \
    [0,0,0,0,0,1],\
    [0,2,0,2,2,0],\
    [0,2,0,0,2,0],\
    [3,2,2,0,2,0],\
    [1,0,2,0,0,1]]           
    # 0 for path with people, 1 for exit, 2 for wall, 3 for fire
    population_map = [  \
    [0,0,0,0,10,0], \
    [0,0,0,0,0,10],\
    [0,0,0,0,0,0],  \
    [0,0,0,0,0,0],  \
    [0,0,0,0,0,0]]
    # number of people in cell
    init_point = [0,0]
    # location of users

    action = [[-1,0], [1,0], [0,-1], [0,1]]
            #[up, down, left, right]
    reward = [path_reward, 100, -99999, -50]

    start = time.time()
    for _ in range(1):
        print(_)
        best_exit,path_to_exit = Q_learning(state_map, population_map, action, reward, init_point)
    print("%.2f second",(time.time() - start)/1000)

    print(best_exit, path_to_exit)

    plt.figure(1)
    for i in path_to_exit:
        present_state_map = [ r[:] for r in state_map]
        present_state_map[i[0]][i[1]] = -1
        plt.imshow(present_state_map, interpolation='none',cmap = "rainbow",alpha = 0.5)
        plt.imshow(population_map, interpolation='none',cmap = "Blues",alpha = 0.5)
        #plt.pause(0.05)
        plt.show()