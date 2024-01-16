import pygame
import torch
from Constants import *
from Environment import Environment
from DQN_Agent import DQN_Agent
from ReplayBuffer import ReplayBuffer

buffer_path = "Data/buffer1.pth"
DQN_path = "Data/DQN1.pth"

def main ():

    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Space')
    # clock = pygame.time.Clock()

    header_surf = pygame.Surface((WIDTH, 100))
    main_surf = pygame.Surface((WIDTH, HEIGHT - 100))
    header_surf.fill(BLUE)
    main_surf.fill(LIGHTGRAY)

    env = Environment(surface=main_surf)

    screen.blit(header_surf, (0,0))
    screen.blit(main_surf, (0,100))
    write (header_surf, "Score: " + str(env.score) + " Ammunition: " + str(env.spaceship.ammunition))

    best_score = 0

    ####### params ############
    player = DQN_Agent()
    player_hat = DQN_Agent()
    player_hat.DQN = player.DQN.copy()
    batch_size = 64
    buffer = ReplayBuffer()
    learning_rate = 0.01
    ephocs = 100000
    C = 10

    optim = torch.optim.Adam(player.DQN.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim,100000, gamma=0.50)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[25000*30, 30*50000, 30*100000, 30*250000, 30*500000], gamma=0.5)

    
    for epoch in range(ephocs):
        env.restart()
        end_of_game = False

        while not end_of_game:
            main_surf.fill(LIGHTGRAY)
            header_surf.fill(BLUE)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return
            
            # Sample Environement
            state = env.state()
            action = player.get_Action(state=state, epoch=epoch)
            reward, done = env.move(action=action)
            next_state = env.state()
            buffer.push(state, torch.tensor(action, dtype=torch.int64), torch.tensor(reward, dtype=torch.float32), 
                        next_state, torch.tensor(done, dtype=torch.float32))
            if done:
                write (header_surf, "Score: " + str (env.score))
                screen.blit(header_surf, (0,0))
                pygame.display.update()
                best_score = max(best_score, env.score)
                break

            state = next_state

            write(header_surf, "Score: " + str(env.score), (200, 60))
            write(header_surf, "Ammunition: " + str(env.spaceship.ammunition),(400, 60))
            write(header_surf,"Level: " + str(env.level), (200, 20))
            write(header_surf, "epoch: " + str (epoch), (400, 20))
            screen.blit(header_surf, (0,0))
            screen.blit(main_surf, (0,100))
            pygame.display.update()
            # clock.tick(FPS)
            
            if len(buffer) < MIN_BUFFER:
                continue
    
            ############## Train ################
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            Q_values = player.Q(states, actions)
            next_actions = player_hat.get_Actions(next_states)
           
            with torch.no_grad():
                Q_hat_Values = player_hat.Q(next_states, next_actions)

            loss = player.DQN.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()

            

        if epoch % C == 0:
            player_hat.DQN.load_state_dict(player.DQN.state_dict())

        #########################################
        if len(buffer) > MIN_BUFFER:
            print (f'epoch: {epoch} loss: {loss} best_score: {best_score}')

        if epoch % 1000:
            torch.save(buffer, buffer_path)
            player.save_param(DQN_path)

        

def write (surface, text, pos = (50, 20)):
    font = pygame.font.SysFont("arial", 36)
    text_surface = font.render(text, True, WHITE, BLUE)
    surface.blit(text_surface, pos)


        
if __name__ == "__main__":
    main ()