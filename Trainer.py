import pygame
import torch
from CONSTANTS import *
from Environment import Environment
from DQN_Agent import DQN_Agent
from ReplayBuffer import ReplayBuffer

# buffer_path = "Data/buffer7.pth"
# DQN_path = "Data/DQN7.pth"
# results_path = "Data/results7.pth"

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
    batch_size = 50
    buffer = ReplayBuffer(path=None)
    learning_rate = 0.00001
    ephocs = 100000
    start_epoch = 0
    C = 10
    loss = torch.tensor(-1)
    scores, losses = [], []
    optim = torch.optim.Adam(player.DQN.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim,100000, gamma=0.50)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[1000*1000, 3000*1000, 5000*1000], gamma=0.5)
    step = 0

    ######### checkpoint ############
    checkpoint_path = "Data/checkpoint1.pth"
    # checkpoint = torch.load(checkpoint_path)
    # start_epoch = checkpoint['epoch']
    # player.DQN.load_state_dict(checkpoint['model_state_dict'])
    # player_hat.DQN.load_state_dict(checkpoint['model_state_dict'])
    # optim.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[1000*1000, 3000*1000, 5000*1000], gamma=0.5)
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # buffer = checkpoint['buffer']
    # losses = checkpoint['losses']
    # scores = checkpoint['scores']
    

    #################################

    for epoch in range(start_epoch, ephocs):
        
        env.restart()
        end_of_game = False
        state = env.state()
        while not end_of_game:
            print (step, end='\r')
            step += 1
            main_surf.fill(LIGHTGRAY)
            header_surf.fill(BLUE)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return
            
            ############## Sample Environement #########################
            action = player.get_Action(state=state, epoch=epoch)
            reward, done = env.move(action=action)
            next_state = env.state()
            buffer.push(state, torch.tensor(action, dtype=torch.int64), torch.tensor(reward, dtype=torch.float32), 
                        next_state, torch.tensor(done, dtype=torch.float32))
            if done:
                best_score = max(best_score, env.score)
                break

            state = next_state

            write(header_surf,"Level: " + str(env.level), (200, 20))
            write(header_surf, "epoch: " + str (epoch), (400, 20))
            write(header_surf, "Score: " + str(env.score), (200, 60))
            write(header_surf, "Ammunition: " + str(env.spaceship.ammunition),(400, 60))
            screen.blit(header_surf, (0,0))
            screen.blit(main_surf, (0,100))
            pygame.display.update()
            # clock.tick(FPS)
            
            if len(buffer) < MIN_BUFFER:
                continue
    
            ############## Train ################
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            Q_values = player.Q(states, actions)
            next_actions, Q_hat_Values = player_hat.get_Actions_Values(next_states)

            loss = player.DQN.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()

        if epoch % C == 0:
            player_hat.DQN.load_state_dict(player.DQN.state_dict())

        #########################################
        print (f'epoch: {epoch} loss: {loss:.7f} LR: {scheduler.get_last_lr()} step: {step} ' \
               f'score: {env.score} level: {env.level} best_score: {best_score}')
        step = 0

        if epoch % 10 == 0:
            scores.append(env.score)
            losses.append(loss.item())

        if epoch % 1000 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': player.DQN.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'buffer': buffer,
                'loss': losses,
                'scores':scores
            }
            torch.save(checkpoint, checkpoint_path)
            # torch.save(buffer, buffer_path)
            # player.save_param(DQN_path)
            # torch.save((scores, losses), results_path)

        

def write (surface, text, pos = (50, 20)):
    font = pygame.font.SysFont("arial", 36)
    text_surface = font.render(text, True, WHITE, BLUE)
    surface.blit(text_surface, pos)


        
if __name__ == "__main__":
    main ()