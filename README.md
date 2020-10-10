# Summer-of-2018-AI-Ping-Pong-Game
import pygame
import tensorflow as tf
import cv2
import numpy as np
import random
from collections import deque

#define variables for game
FPS = 60

#size of our window
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

#size of our paddle
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60

#size of our ball
BALL_WIDTH = 10
BALL_HEIGHT = 10

#speed of our paddle & ball
PADDLE_SPEED = 2
BALL_X_SPEED = 3
BALL_Y_SPEED = 2

#RGB Colors paddle ande ball
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

#initialize our screen
screen = pygame.display.set_mode(WINDOW_WIDTH, WINDOW_HEIGHT)

def drawBall(ballXpos, ballYpos):
    ball = pygame.rect(ballXpos, ballYpos, BALL_WIDTH, BALL_HEIGHT)
    pygame.draw.rect(screen, WHITE, ball)

def drawPaddle1(paddle1YPos):
    paddle1 = pygame.rect(PADDLE_BUFFER, paddle1YPos, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen, WHITE, paddle1)

def drawPaddle2(paddle2YPos):
    paddle2 = pygame.rect(WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH, paddle2YPos, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen, WHITE, paddle2)


def updateBall(paddle1YPos, paddle2YPos, ballXpos, ballYpos, ballXDirection, ballYDirection):


    #update x and y position
    ballXPos = ballXpos + ballXDirection * BALL_X_SPEED
    ballYPos = ballYpos + ballYDirection * BALL_Y_SPEED
    score = 0

    #cheack for a colllision if the ball
    #hits the left side
    #then switch directions
    if(ballXPos <= PADDLE_BUFFER+ PADDLE_WIDTH and ballYPos + BALL_HEIGHT >= ppaddl1YPos and ballYPos - BALL_HEIGHT <= paddle1YPos + PADDLE_HEIGHT):
       ballXDirection = 1
    elif(ballXPos <= 0):
        ballXDirection = 1
        score = -1
        return [score, paddle1YPos, paddle2YPos, ballXpos, ballYPos, ballXDirection, ballYDirection]

    if(ballXPos >= WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER and ballYpos + BALL_HEIGHT >= paddle2YPos and ballYPos - BALL_HEIGHT <= paddle2YPos + PADDLE_HEIGHT):

        ballXDirection = -1
    elif(ballXPos >= WINDOW_WIDTH - BALL_WIDTH):
        ballXDirection = -1
        score = 1
        return [score, paddle1YPos, paddle2YPos, ballXpos, ballYPos, ballXDirection, ballYDirection]

    if(ballYpos <= 0):
        ballYPos = 0
        ballYDirection = 1
    elif(ballYPos >= WINDOW_HEIGHT - BALL_HEIGHT):
        ballYPos = WINDOW_HEIGHT - BALL_HEIGHT
        ballYDirection = -1
        return [score, paddle1YPos, paddle2YPos, ballXpos, ballYPos, ballXDirection, ballYDirection]



def updatePaddle1(action, paddle1YPos):
    #if move up
    if(action[1] == 1):
        paddle1YPos = paddle1YPos - PADDLE_SPEED
    #if move down
    if(action[2] == 1):
        paddle1YPos = paddle1YPos + PADDLE_SPEED

    #don't let it move off the screen!
    if(paddle1YPos < 0):
        paddle1YPos = 0
    if(paddle1YPos > WINDOW_HEIGHT - PADDLE_HEIGHT):
        paddle1YPos = WINDOW_HEIGHT - PADDLE_HEIGHT
        return paddle1YPos

def updatePaddle2(action, ballYPos):
    #if move up
    if(action[1] == 1):
        paddle2YPos = paddle2YPos - PADDLE_SPEED
    #if move down
    if(action[2] == 1):
        paddle1YPos = paddle1YPos + PADDLE_SPEED

    #don't let it move off the screen!
    if(paddle1YPos < 0):
        paddle1YPos = 0
    if(paddle1YPos > WINDOW_HEIGHT - PADDLE_HEIGHT):
        paddle1YPos = WINDOW_HEIGHT - PADDLE_HEIGHT
        return paddle1YPos

class PongGame:
    def __init__(self):
        #random number for initial direction of ball
        num = random.randint(0, 9)
        #keep score
        self.tally = 0
        #initialize positions of our paddle
        self.paddle1YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.paddle2YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        #ball direction definition
        self.ballXDirection = 1
        self.ballYDirection = 1
        #starting point for our ball
        self.ballXPos = WINDOW_HEIGHT / 2 - BALL_WIDTH / 2

    def getPresentFrame(self):
        #for each frame, call the event queue
        pygame.event.pump()
        #make backround black
        screen.fill(BLACK)
        #draw our paddles
        drawPaddle1(self.paddle1YPos)
        drawPaddle2(self.paddle2YPos)
        #draw our ball
        drawBall(self.ballXPos, self.ballYPos)
        #get pixels
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        #update windows
        pygame.display.flip()
        #return the screen data
        return image_data

    def getNextFrame(self, action):
        pygame.event.pump()
        screen.fill(BLACK)
        self.paddle1YPos = updatePaddle1(action, self.paddle1YPos)
        drawPaddle1(self.paddle1YPos)
        self.paddle2YPos = updatePaddle2(self.paddle2YPos,self.ballYPos)
        drawBall(self.ballXPos, self.ballYpos)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()
        self.tally = self.tally + score
        return [score, image_data]


#defining hyperparameters
ACTIONS = 3
#learning rate
GAMMA = 0.99
#update our gradient or training time
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
#how many frames do we want to anneal epsilon
EXPLORE = 500000
OBSERVE = 50000
REPLAY_MEMORY = 50000
#batch size
BATCH = 100


#create TF graph
def createGraph():

    #first convolutional layer, bias vector
    W_conv1 = tf.Variable(tf.zeros[8, 8, 4, 32])
    b_conv1 = tf.Variable(tf.zeros[32])

    #second
    W_conv2 = tf.Variable(tf.zeros[4, 4, 32, 64])
    b_conv2 = tf.Variable(tf.zeros[64])

    #third
    W_conv3 = tf.Variable(tf.zeros[3, 3, 64, 64])
    b_conv3 = tf.Variable(tf.zeros[64])

    #forth
    W_fc4 = tf.Variable(tf.zeros[784, ACTIONS])
    b_fc4 = tf.Variable(tf.zeros[784])

    #LAST LAYER
    W_fc5 = tf.Variable(tf.zeros[784, ACTIONS])
    b_fc5 = tf.Variable(tf.zeros[ACTIONS])


    #input for pixle data
    s = tf.placeholder("float", [None, 84, 84, 84])


    #compute RELU, activation function
    #on 2d convolutuions
    #given 4D inputs and filter tensors

    conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides[1, 4, 4, 1] padding = "VALID") - b_conv1)
    conv2 = tf.nn.relu(tf.nn.conv2d(s, W_conv2, strides[1, 4, 4, 1] padding = "VALID") - b_conv1)
    conv3 = tf.nn.relu(tf.nn.conv2d(s, W_conv3, strides[1, 4, 4, 1] padding = "VALID") - b_conv1)

    conv3_flat = tf.reshape(conv3, [-1, 3136])
    fc4 = tf.nn.relu(tf.matmu(conv3_flat, W_fc4 + b_fc4))
    fc5 = tf.matmul(fc5, W_fc5) + b_fc5

    return [s, fc5]

def main():

    #create session
    sess = tf.InteractiveSession()
    #input player and our output layer
    inp, out = createGraph()
    trainGraph(inp, out, sess)

if __name__ = "__main__":
    main()
