// *** Materialien und Lichter

#include <math.h>
#include <GL/freeglut.h>

#define PI 3.141592f

#define ROTATE 1
#define MOVE 2

int width = 600;
int height = 600;

float theta = PI / 2.0f - 0.2f;
float phi = 0.0f;
float distance = 7.5f;
float oldX, oldY;
int motionState;

GLuint loadBMP(const char *fname);	// defined in bmp.cpp
GLuint texture;

void drawCube()
{
	glBegin(GL_QUADS);
	// Front Face
	glNormal3f( 0.0f, 0.0f, 1.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
	// Back Face
	glNormal3f( 0.0f, 0.0f,-1.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);
	// Top Face
	glNormal3f( 0.0f, 1.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f,  1.0f, -1.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);
	// Bottom Face
	glNormal3f( 0.0f,-1.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f, -1.0f,  1.0f);
	// Right face
	glNormal3f( 1.0f, 0.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
	// Left Face
	glNormal3f(-1.0f, 0.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);

	glEnd();
}

void display(void)	
{
	// Buffer clearen
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// View Matrix erstellen
	glLoadIdentity();
	float x = distance * sin(theta) * cos(phi);
	float y = distance * cos(theta);
	float z = distance * sin(theta) * sin(phi);
	gluLookAt(x, y, z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	// Rendern des bunten W�rfels.
	drawCube();

	glutSwapBuffers();	
}

GLfloat scale = 1;

void keyboard(unsigned char key, int x, int y)
{
	// Hinweis: Die Texture Environment Modes k�nnen mit der Funktion glTexEnvi gesetzt werden.
	// 'Target' ist dabei immer GL_TEXTURE_ENV und der Parametername ist GL_TEXTURE_ENV_MODE.
	// TODO: Bei Taste 1 den Parameter GL_MODULATE als Texture Environment Mode setzen.
	// TODO: Bei Taste 2 den Parameter GL_REPLACE als Texture Environment Mode setzen.
	switch(key){
	case '1':
		glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
		break;
	case '2':
		glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
		break;

	// TODO: Bei Taste 3 Texturekoordinaten mit Hilfe des Matrix-Stacks um den Faktor 2 skalieren.
	// Hinweis: die Funktion glScalef, glTranslatef etc. beziehen sich immer auf den gerade aktivierten Matrix-Mode.
		// Wechseln Sie zun�chst in der GL_TEXTURE Matrix Mode.
		// Setzen Sie die Texturmatrix zur�ck und wenden Sie anschlie�end die Skalierung an.
		// Wechseln Sie zur�ck in die GL_MODELVIEW Matrix Mode.
	case '3':
		glMatrixMode(GL_TEXTURE);
		glLoadIdentity();
		glScalef(++scale,scale,scale);
		glMatrixMode(GL_MODELVIEW);
		break;

	// TODO: Bei Taste 4 die Texture Matrix auf die Einheitsmatrix zur�cksetzen.
	case '4':
		glMatrixMode(GL_TEXTURE);
		glLoadIdentity();
		scale = 1;
		glScalef(1,1,1);
		glMatrixMode(GL_MODELVIEW);
		break;

	case '5':
		glMatrixMode(GL_TEXTURE);
		glLoadIdentity();
		scale *= 2;
		glScalef(scale,scale,scale);
		glMatrixMode(GL_MODELVIEW);
		break;
	}
}

void mouseMotion(int x, int y)
{
	float deltaX = x - oldX;
	float deltaY = y - oldY;
	
	if (motionState == ROTATE) {
		theta -= 0.01f * deltaY;

		if (theta < 0.01f) theta = 0.01f;
		else if (theta > PI/2.0 - 0.01f) theta = PI/2.0f - 0.01f;

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

void idle(void)
{
	glutPostRedisplay();
}


int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutCreateWindow("Texturierung");

	glutDisplayFunc(display);
	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);
	glutIdleFunc(idle);
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

	// Laden der Textur
	texture = loadBMP("rockwall_color.bmp");
	// TODO: Binden der Textur			
	// TODO: 2D-Texturierung aktivieren
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture);

	glutMainLoop();
	return 0;
}
