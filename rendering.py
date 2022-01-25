from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *

def render_dots(Points):
    pygame.init()
    display=(1600,1600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 1000.0)
    glTranslatef(0.0, 0.0, -100)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # glRotatef(1, 3, 1, 1)
        glColor3f(1, 0, 0)
        glPointSize(10)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Dots(Points)
        glBegin(GL_POINTS)
        # import pdb; pdb.set_trace()
        for point in range(len(Points)):
            # print(point)
            tmp = Points[point]
            glVertex3f(tmp[0],tmp[1],tmp[2])
        glEnd()

        pygame.display.flip()
        pygame.time.wait(1000)

def render_dots_and_lines(Points, topology):
    pygame.init()
    display=(1600,1600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 1000.0)
    glTranslatef(0.0, 0.0, -200)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # glRotatef(1, 3, 1, 1)
        glColor3f(1, 0, 0)
        glPointSize(10)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Dots(Points)
        glBegin(GL_POINTS)        
        for point in range(len(Points)):
            # print(point)
            tmp = Points[point]
            glVertex3f(tmp[0],tmp[1],tmp[2])
        glEnd()

        # draw lines 
        glBegin(GL_LINES)        
        for point in range(1, len(Points)):
            glColor3f(1, point/len(Points), 0)
            parent = topology[point]
            parentVertex = Points[parent]
            currentVertex = Points[point]

            glVertex3f(parentVertex[0], parentVertex[1], parentVertex[2])
            glVertex3f(currentVertex[0], currentVertex[1], currentVertex[2])
        glEnd()

        pygame.display.flip()
        pygame.time.wait(1000)

if __name__ == "__main__":
    Points = []
    # Points.append([0.5, 0.5, 0])
    # Points.append([1.0, 1.0, 0])

    # Points.append([-0.5, -0.5, 0])
    # Points.append([-1.0, -1.0, 0])
    render_dots(Points)
