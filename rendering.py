from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *

def render_dots(Points):
    pygame.init()
    display=(1600,1600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -10)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # glRotatef(1, 3, 1, 1)
        glColor3f(1, 0, 0)
        glPointSize(5)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Dots(Points)
        glBegin(GL_POINTS)
        for point in range(len(Points)):
            # print(point)
            tmp = Points[point]
            glVertex3f(tmp[0],tmp[1],tmp[2])
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
