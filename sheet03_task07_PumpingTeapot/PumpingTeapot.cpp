// Framework f�r GLSL-Programme
// Pumping and Glowing Teapot 

#include <GL/glew.h>
#include <stdlib.h> /* Known bug in GLUT, see http://www.lighthouse3d.com/opengl/glut/ (bottom of page) for an explanation */
#include <GL/freeglut.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

// Global variables
GLfloat alpha = 0;

// GLSL related variables
// Pumping Shader Program
GLuint vertexShaderPumping = -1;	
GLuint fragmentShaderPumping = -1;
GLuint shaderProgramPumping = -1;

// Window size
int width = 512;       
int height = 512;

// uniform locations
GLint uniformTime;

bool usePumping = true;

// Print information about the compiling step
void printShaderInfoLog(GLuint shader)
{
	if (shader == -1)
		return;
    GLint infologLength = 0;
    GLsizei charsWritten  = 0;
    char *infoLog;

	glGetShaderiv(shader, GL_INFO_LOG_LENGTH,&infologLength);		
	infoLog = (char *)malloc(infologLength);
	glGetShaderInfoLog(shader, infologLength, &charsWritten, infoLog);
	printf("%s\n",infoLog);
	free(infoLog);
}

// Print information about the linking step
void printProgramInfoLog(GLuint program)
{
	if (program == -1)
		return;

	GLint infoLogLength = 0;
	GLsizei charsWritten  = 0;
	char *infoLog;

	glGetProgramiv(program, GL_INFO_LOG_LENGTH,&infoLogLength);
	infoLog = (char *)malloc(infoLogLength);
	glGetProgramInfoLog(program, infoLogLength, &charsWritten, infoLog);
	printf("%s\n",infoLog);
	free(infoLog);
}

// Reads a file and returns the content as a string
string readFile(string fileName)
{
	string fileContent;
	string line;

	ifstream file(fileName.c_str());
	if (file.is_open()) {
		while (!file.eof()){
			getline (file,line);
			line += "\n";
			fileContent += line;					
		}
		file.close();
	}
	else
		cout << "ERROR: Unable to open file " << fileName << endl;

	return fileContent;
}

void initGL()
{
   // Initialize camera
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   gluPerspective(45, 1, 0.1, 100);
   glMatrixMode(GL_MODELVIEW);

   // Initialize light source
   GLfloat light_pos[] = {10, 10, 10, 1};
   GLfloat light_col[] = { 1,  1,  1, 1};

   glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
   glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_col);
   glLightfv(GL_LIGHT0, GL_SPECULAR, light_col);

   // Enable lighting
   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);

   // Initialize material
   GLfloat teapot_diffuse[]  = {0.75f, 0.375f, 0.075f, 1};
   GLfloat teapot_specular[] = {0.8f, 0.8f, 0.8f, 1};

   glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, teapot_diffuse);
   glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, teapot_specular);
   glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 45.2776f);

   // Enable depth buffer
   glEnable(GL_DEPTH_TEST);
}

void initGLSL()
{
	// TODO: Create empty shader object (vertex shader) and assign it to 'vertexShaderPumping'
	
	// Read vertex shader source 
	string shaderSource = readFile("pumping.vert");
	const char* sourcePtr = shaderSource.c_str();

	// TODO: Attach shader code
	
	// TODO: Compile shader	
	
	printShaderInfoLog(vertexShaderPumping);

	// TODO: Create empty shader object (fragment shader) and assign it to 'fragmentShaderPumping'

	// Read vertex shader source 
	shaderSource = readFile("pumping.frag");
	sourcePtr = shaderSource.c_str();

	// TODO: Attach shader code

	// TODO: Compile shader

	printShaderInfoLog(fragmentShaderPumping);

	// TODO: Create shader program and assign it to 'shaderProgramPumping'

	// TODO: Attach shader vertex shader and fragment shader to program	

	// TODO: Link program
	
	printProgramInfoLog(shaderProgramPumping);

	// TODO: Use program.	

	// TODO: Teilaufgabe 3... Die Uniform Location der Zeit-Variable bestimmen.	
}

void keyboard(unsigned char key, int x, int y)
{
	// set parameters
	switch (key) 
	{                                 
		case 'p':
			usePumping = !usePumping;
			break;
	}
}

void display()
{	
	// Pumping Shader anschalten falls aktiviert
	if (usePumping) {
		glUseProgram( shaderProgramPumping );
		// TODO: Den Zeitparameter (uniform) aktualisieren.
	}
	else {
		glUseProgram( 0 );
	}

	// Clear window
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLoadIdentity();
	gluLookAt(10, 7, 10, 0, 0, 0, 0, 1, 0);

	glRotatef(alpha, 0, 1, 0);
	glutSolidTeapot(3);

	// Increment rotation angle
	alpha += 1;

	// TODO: Inkrementieren des Zeit Parameters.

	// Swap display buffers
	glutSwapBuffers();
}

void timer(int value)
{
   // Call timer() again in 25 milliseconds
   glutTimerFunc(25, timer, 0);

   // Redisplay frame
   glutPostRedisplay();
}

int main(int argc, char** argv)
{
   // Initialize GLUT
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
   glutInitWindowSize(width, height);
   glutCreateWindow("Pumping Teapot");

   // Init glew so that the GLSL functionality will be available
   if(glewInit() != GLEW_OK)
	   cout << "GLEW init failed!" << endl;

	// OpenGL/GLSL initializations
	initGL();
	initGLSL();

	// Register callback functions   
	glutKeyboardFunc(keyboard);
	glutDisplayFunc(display);
	glutTimerFunc(25, timer, 0);     // Call timer() in 25 milliseconds

	// Enter main loop
	glutMainLoop();

	return 0;
}
