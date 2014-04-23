// *** Spiegelungen mit Stencil Buffer simulieren

#include <math.h>
#include <GL/freeglut.h>	

GLfloat viewPos[3] = {0.0f, 2.0f, 2.0f};

#define PI 3.141592f

#define ROTATE 1
#define MOVE 2

int width = 600;
int height = 600;

float theta = PI / 2.0f - 0.01f;
float phi = 0.0f;
float distance = 2.5f;
float oldX, oldY;
int motionState;

float angle_x_mirror = 0;
// dreht spiegel um Normale
float angle_y_mirror = 0;
float angle_z_mirror = 0;

float pos_mirror[3] = {2,-3,7};

GLfloat lightPos[4] = {3, 3, 3, 1};
GLfloat mirrorColor[4] = {1.0f, 0.2f, 0.2f, 0.5f};
GLfloat teapotColor[4] = {0.8f, 0.8f, 0.2f, 1.0f};


// Szene zeichnen: Eine Teekanne
void drawScene()
{
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, teapotColor);
	glPushMatrix();
	glTranslatef(0,0.37f,0);
	glutSolidTeapot(0.5f);
	glPopMatrix();
}

// Spiegel zeichen: Ein Viereck
void drawMirror(GLint size)
{
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mirrorColor);
	glPushMatrix();
	glRotatef(angle_x_mirror,1,0,0);
	glRotatef(angle_z_mirror,0,0,1);
	glRotatef(angle_y_mirror,0,1,0);
	glTranslatef(pos_mirror[0],pos_mirror[1],pos_mirror[2]);
	glBegin(GL_QUADS);
	glVertex3f(size,0,size);
	glVertex3f(size,0,-1*size);
	glVertex3f(-1*size,0,-1*size);
	glVertex3f(-1*size,0,size);
	glEnd();
	glPopMatrix();
}

void display(void)	
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	glLoadIdentity();
	float x = distance * sin(theta) * cos(phi);
	float y = distance * cos(theta);
	float z = distance * sin(theta) * sin(phi);

	gluLookAt(x, y, z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	// Szene normal zeichnen (ohne Spiegelobjekt)

	glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
	drawScene();

	// *** Spiegel zeichnen, so dass Spiegelobjekt im Stencil Buffer eingetragen wird
	// *** Framebuffer dabei auf Read-Only setzen, Depth Buffer deaktivieren, Stencil Test aktivieren
	glClear(GL_STENCIL_BUFFER_BIT);
	glStencilFunc(GL_ALWAYS,1,1);
	glStencilOp(GL_REPLACE,GL_REPLACE,GL_REPLACE);
	glEnable(GL_STENCIL_TEST);
	glDisable(GL_DEPTH_TEST);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	drawMirror(5);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	//glDisable(GL_STENCIL_TEST);
	glEnable(GL_DEPTH_TEST);

	// *** Gespiegelte Szene zeichnen, Stencil Buffer so einstellen, dass nur bei
	// *** einem Eintrag 1 im Stencil Buffer das entsprechende Pixel im Framebuffer 
	// *** gezeichnet wird, der Inhalt vom Stencil Buffer soll unveraendert bleiben 
	// *** Depth Buffer wieder anmachen, Framebuffer Maskierung deaktivieren
	// *** Was macht man mit der Lichtquelle ?
	glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP);
	glStencilFunc(GL_EQUAL,1,1);
	
	glPushMatrix();
	glRotatef(angle_x_mirror,1,0,0);
	glRotatef(angle_z_mirror,0,0,1);
	// Translation entlang der Spiegelnormalen
	glTranslatef(0,2*pos_mirror[1],0);
	glRotatef(angle_z_mirror,0,0,1);
	glRotatef(angle_x_mirror,1,0,0);
	glScalef(1,-1,1);
	glLightfv(GL_LIGHT0,GL_POSITION, lightPos);
	drawScene();
	glPopMatrix();

	glLightfv(GL_LIGHT0,GL_POSITION, lightPos);
	glDisable(GL_STENCIL_TEST);
	// *** Stencil Test deaktivieren, Spiegelung der Szene rueckgaengig machen
	// *** Spiegelobjekt mit diffuser Farbe mirrorColor zeichen
	// *** Blending aktivieren und ueber Alpha-Kanal mit Spiegelbild zusammenrechnen
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	drawMirror(5);
	glDisable(GL_BLEND);

	glutSwapBuffers();	
}

void mouseMotion(int x, int y)
{
	float deltaX = x - oldX;
	float deltaY = y - oldY;
	
	if (motionState == ROTATE) {
		theta -= 0.01f * deltaY;

		if (theta < 0.01f) theta = 0.01f;
		else if (theta > PI/2.0f - 0.01f) theta = PI/2.0f - 0.01f;

		phi += 0.01f * deltaX;	
		if (phi < 0) phi += 2*PI;
		else if (phi > 2*PI) phi -= 2*PI;
	}
	else if (motionState == MOVE) {
		distance += 0.01f * deltaY;
	}

	oldX = (float)x;
	oldY = (float)y;

	glutPostRedisplay();

}

void mouse(int button, int state, int x, int y)
{
	oldX = (float)x;
	oldY = (float)y;

	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) {
			motionState = ROTATE;
		}
	}
	else if (button == GLUT_RIGHT_BUTTON) {
		if (state == GLUT_DOWN) {
			motionState = MOVE;
		}
	}
}

void keyboard(unsigned char key, int x, int y)
{
	switch(key)
	{
	case '1':
		angle_x_mirror += 1;
		break;
	case '2':
		angle_z_mirror += 1;
		break;
	case '3':
		angle_x_mirror = 0;
		break;
	case '4':
		angle_z_mirror = 0;
		break;
	case '5':
		pos_mirror[0] += 0.2;
		break;
	case '6':
		pos_mirror[0] -= 0.2;
		break;
	case '7':
		pos_mirror[1] -= 0.2;
		break;
	case '8':
		pos_mirror[2] += 0.2;
		break;
	case '9':
		pos_mirror[2] -= 0.2;
		break;
	case '0':
		pos_mirror[0] = 0;
		pos_mirror[1] = 0;
		pos_mirror[2] = 0;
		break;
	}
	glutPostRedisplay();
}



void idle(void)
{
	glutPostRedisplay();
}


int main(int argc, char **argv)
{
	GLfloat fogColor[] = {0.5f, 0.5f, 0.5f, 1.0f};
	GLfloat lightPos[4] = {3, 3, 3, 1};

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
	glutInitWindowSize(width, height);
	glutCreateWindow("Teapot im Spiegel");

	glutDisplayFunc(display);
	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);
	glutKeyboardFunc(keyboard);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_DEPTH_TEST);

	glViewport(0,0,width,height);					
	glMatrixMode(GL_PROJECTION);					
	glLoadIdentity();								

	gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glutMainLoop();
	return 0;
}
