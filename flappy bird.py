import pygame
import random
import numpy as np
import sys

# Initialize pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.5
JUMP_STRENGTH = -7
PIPE_GAP = 150
PIPE_WIDTH = 60
GROUND_HEIGHT = 100
SPEED = 3
FONT = pygame.font.SysFont('Arial', 20)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
SKY_BLUE = (135, 206, 235)

# Set up the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Genetic Flappy Bird')
clock = pygame.time.Clock()

class Bird:
    def __init__(self, brain=None):
        self.reset()
        
        # Neural network: simple with 3 inputs, 4 hidden neurons, 1 output
        if brain is None:
            self.brain = {
                'weights1': np.random.randn(3, 4) * 0.1,
                'weights2': np.random.randn(4, 1) * 0.1,
                'bias1': np.zeros((1, 4)),
                'bias2': np.zeros((1, 1))
            }
        else:
            self.brain = brain
    
    def reset(self):
        self.x = 80
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.radius = 15
        self.dead = False
        self.score = 0
        self.lifetime = 0
        self.fitness = 0
    
    def jump(self):
        self.velocity = JUMP_STRENGTH
    
    def update(self):
        if self.dead:
            return
        
        self.velocity += GRAVITY
        self.y += self.velocity
        
        # Check boundaries
        if self.y - self.radius <= 0:
            self.y = self.radius
            self.velocity = 0
        
        if self.y + self.radius >= SCREEN_HEIGHT - GROUND_HEIGHT:
            self.dead = True
            self.y = SCREEN_HEIGHT - GROUND_HEIGHT - self.radius
            self.velocity = 0
        
        self.lifetime += 1
    
    def think(self, pipes):
        closest_pipe = None
        closest_dist = float('inf')
        
        for pipe in pipes:
            if pipe.x + PIPE_WIDTH > self.x and pipe.x < closest_dist:
                closest_pipe = pipe
                closest_dist = pipe.x
        
        if closest_pipe is None:
            return False
        
        inputs = np.array([
            self.y / SCREEN_HEIGHT,
            (closest_pipe.x - self.x) / SCREEN_WIDTH,
            (closest_pipe.gap_y) / SCREEN_HEIGHT
        ])
        
        hidden = np.tanh(np.dot(inputs, self.brain['weights1']) + self.brain['bias1'])
        output = np.tanh(np.dot(hidden, self.brain['weights2']) + self.brain['bias2'])
        
        return output[0][0] > 0
    
    def draw(self):
        if not self.dead:
            pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), self.radius)

class Pipe:
    def __init__(self):
        self.x = SCREEN_WIDTH
        self.gap_y = random.randint(150, SCREEN_HEIGHT - GROUND_HEIGHT - 150)
        self.passed = False
    
    def update(self):
        self.x -= SPEED
    
    def is_off_screen(self):
        return self.x + PIPE_WIDTH < 0
    
    def check_passed(self, bird):
        if not self.passed and self.x + PIPE_WIDTH < bird.x:
            self.passed = True
            return True
        return False
    
    def check_collision(self, bird):
        if bird.dead:
            return False
        
        if bird.x + bird.radius > self.x and bird.x - bird.radius < self.x + PIPE_WIDTH:
            if bird.y - bird.radius < self.gap_y - PIPE_GAP//2 or bird.y + bird.radius > self.gap_y + PIPE_GAP//2:
                return True
        return False
        
    def draw(self):
        pygame.draw.rect(screen, GREEN, (self.x, 0, PIPE_WIDTH, self.gap_y - PIPE_GAP//2))
        bottom_pipe_top = self.gap_y + PIPE_GAP//2
        pygame.draw.rect(screen, GREEN, (self.x, bottom_pipe_top, PIPE_WIDTH, SCREEN_HEIGHT - bottom_pipe_top - GROUND_HEIGHT))

class GeneticAlgorithm:
    def __init__(self, population_size=50):
        self.population_size = population_size
        self.birds = [Bird() for _ in range(population_size)]
        self.pipes = []
        self.next_pipe_timer = 0  # Start immediately
        self.generation = 1
        self.best_score = 0
        self.alive_count = population_size
    
    def update(self):
        # Pipe generation
        self.next_pipe_timer -= 1
        if self.next_pipe_timer <= 0:
            self.pipes.append(Pipe())
            self.next_pipe_timer = 100  # Reset to original frequency
        
        # Update pipes
        for pipe in self.pipes[:]:
            pipe.update()
            if pipe.is_off_screen():
                self.pipes.remove(pipe)
        
        # Update birds
        self.alive_count = 0
        for bird in self.birds:
            if not bird.dead:
                self.alive_count += 1
                if bird.think(self.pipes):
                    bird.jump()
                bird.update()
                
                for pipe in self.pipes:
                    if pipe.check_collision(bird):
                        bird.dead = True
                        break
                    if pipe.check_passed(bird):
                        bird.score += 1
                        self.best_score = max(self.best_score, bird.score)
        
        if self.alive_count == 0:
            self.create_next_generation()
    
    def calculate_fitness(self):
        for bird in self.birds:
            bird.fitness = bird.lifetime + (bird.score ** 2) * 100
    
    def create_next_generation(self):
        self.calculate_fitness()
        
        # Normalize fitness
        total_fitness = sum(bird.fitness for bird in self.birds)
        if total_fitness > 0:
            for bird in self.birds:
                bird.fitness /= total_fitness
        
        # Create new generation
        new_birds = []
        
        # Elite selection
        sorted_birds = sorted(self.birds, key=lambda x: x.fitness, reverse=True)
        elite_count = max(2, self.population_size // 10)
        
        # Clone elites with reset state
        for i in range(elite_count):
            elite = Bird(self.copy_brain(sorted_birds[i].brain))
            elite.reset()
            new_birds.append(elite)
        
        # Create rest of population
        while len(new_birds) < self.population_size:
            parent1 = self.select_bird()
            parent2 = self.select_bird()
            child_brain = self.crossover(parent1.brain, parent2.brain)
            child_brain = self.mutate(child_brain)
            child = Bird(child_brain)
            child.reset()
            new_birds.append(child)
        
        # Reset game state
        self.birds = new_birds
        self.pipes = []
        self.next_pipe_timer = 0  # Immediate pipe spawn
        self.generation += 1
    
    def select_bird(self):
        tournament = random.sample(self.birds, 3)
        return max(tournament, key=lambda bird: bird.fitness)
    
    def copy_brain(self, brain):
        return {
            'weights1': np.copy(brain['weights1']),
            'weights2': np.copy(brain['weights2']),
            'bias1': np.copy(brain['bias1']),
            'bias2': np.copy(brain['bias2'])
        }
    
    def crossover(self, brain1, brain2):
        child = {}
        for key in brain1:
            mask = np.random.random(brain1[key].shape) < 0.5
            child[key] = np.where(mask, brain1[key], brain2[key])
        return child
    
    def mutate(self, brain, mutation_rate=0.1):
        for key in brain:
            mutations = np.random.random(brain[key].shape) < mutation_rate
            brain[key] += np.where(mutations, np.random.randn(*brain[key].shape) * 0.5, 0)
        return brain
    
    def draw(self):
        screen.fill(SKY_BLUE)
        pygame.draw.rect(screen, GREEN, (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT))
        
        for pipe in self.pipes:
            pipe.draw()
        
        for bird in self.birds:
            bird.draw()
        
        current_best = max((bird.score for bird in self.birds if not bird.dead), default=0)
        stats = [
            f"Generation: {self.generation}",
            f"Alive: {self.alive_count}/{self.population_size}",
            f"High Score: {self.best_score}",
            f"Current Best: {current_best}"
        ]
        
        for i, text in enumerate(stats):
            surface = FONT.render(text, True, BLACK)
            screen.blit(surface, (10, 10 + i * 25))

def main():
    game = GeneticAlgorithm(population_size=50)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        game.update()
        game.draw()
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()