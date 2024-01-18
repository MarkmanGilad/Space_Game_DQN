import pygame
from CONSTANTS import *
import random
from Bullet import Bullet

class Enemy (pygame.sprite.Sprite):
    shoots_factor = 2
    speed_add = 0
    speed_y = 40
    def __init__(self, img, pos, bullets_Group) -> None:
        super().__init__()
        self.image = img
        self.rect = self.image.get_rect(topleft = pos)
        self.mask = pygame.mask.from_surface(self.image)
        self.speed_x = 5
        self.speed_dir = 1
        self.bullets_Group = bullets_Group

    def update(self) -> None:
        self.move()
        self.shoot()

    def move (self):
        if self.rect.right > WIDTH:
            self.rect.y += Enemy.speed_y
            self.speed_dir = -self.speed_dir
        if self.rect.left < 0:
            self.rect.y += Enemy.speed_y
            self.speed_dir = -self.speed_dir
        self.rect.x += (self.speed_x + Enemy.speed_add) * self.speed_dir

    def shoot (self):
        if random.random() < Enemy.shoots_factor/1000 and len(self.bullets_Group) < MAX_ENEMY_BULLETS:
            self.bullets_Group.add(Bullet(self.rect.midbottom,speed_y=ENEMY_BULLET_SPEED))

    

    

    
        
