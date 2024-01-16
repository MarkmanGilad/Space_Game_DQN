import pygame
from CONSTANTS import *
import random
from Bullet import Bullet

class Enemy (pygame.sprite.Sprite):
    shoots_factor = 0.002
    speed_x = 5
    speed_y = 40
    
    def __init__(self, img, pos, bullets_Group) -> None:
        super().__init__()
        self.image = img
        self.rect = self.image.get_rect(topleft = pos)
        self.mask = pygame.mask.from_surface(self.image)
        self.dir_x = 1
        self.bullets_Group = bullets_Group

    def update(self) -> None:
        self.move()
        self.shoot()

    def move (self):
        self.rect.x += self.speed_x * self.dir_x
        if self.rect.right > WIDTH:
            self.rect.y += Enemy.speed_y
            self.dir_x = -self.dir_x
        if self.rect.left < 0:
            self.rect.y += Enemy.speed_y
            self.dir_x = -self.dir_x

    def shoot (self):
        if random.random() < Enemy.shoots_factor and len(self.bullets_Group) < MAX_ENEMY_BULLETS:
            self.bullets_Group.add(Bullet(self.rect.midbottom,speed_y=ENEMY_BULLET_SPEED))

    

    

    
        
