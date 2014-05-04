// Framework für GLSL-Programme
// Pumping and Glowing Teapot

#include <GL/glew.h>
#include <stdlib.h>
#include <GL/freeglut.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdio>

using namespace std;

// Global variables
GLfloat alpha = 0;

// GLSL related variables
// Blur Shader Program
GLuint vertexShaderBlur = -1;
GLuint fragmentShaderBlur = -1;
GLuint shaderProgramBlur = -1;

// Texture Ids and Framebuffer Object Ids
GLuint teapotTextureId = 0;
GLuint depthTextureId = 0;
GLuint teapotFB = 0;

// Window size
int width = 512;
int height = 512;

// uniform locations
GLint teapotTextureLocation;
GLint blurHorizontalTextureLocation;

bool useBlur = true;

// Print information about the compiling step
void printShaderInfoLog(GLuint shader, int line)
{
	if (shader == -1){
		std::cout << line << ": "<< "shader is -1\n";
		return;
	}

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
void printProgramInfoLog(GLuint program, int line)
{
	if (program == -1){
		std::cout << line << ": "<< "programm is -1\n";
		return;
	}

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



GLuint createShader(string sourceFileName, GLenum type, int line)
{
	// TODO: Create empty shader object
	GLuint shader = glCreateShader(type);
	// Read vertex shader source
	string shaderSource = readFile(sourceFileName);
	char const *shaderSourcePtr = shaderSource.c_str();
	// TODO: Attach shader code
	glShaderSource(shader, 1, &shaderSourcePtr, NULL);
	// TODO: Compile shader
	glCompileShader(shader);

	printShaderInfoLog(shader, line);

	return shader;
}

GLuint createProgram(GLuint vertexShader, GLuint fragmentShader, int line){
	GLuint program = glCreateProgram();

	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);

	glLinkProgram(program);

	printProgramInfoLog(program, line);

	return program;
}

void setUniformFloat(GLuint shader, string name, float value){
	GLint location = glGetUniformLocation(shader, name.c_str());

	if(location == -1){
		cout << "ERROR: No such uniform variable \"" << name
		     << "\" in shader " << shader << endl;
	}
	else{
		glUniform1f(location, value);
	}
}

void setUniformInt(GLuint shader, string name, int value){
	GLint location = glGetUniformLocation(shader, name.c_str());

	if(location == -1){
		cout << "ERROR: No such uniform variable \"" << name
		     << "\" in shader " << shader << endl;
	}
	else{
		glUniform1i(location, value);
	}
}

void initGLSL()
{

	vertexShaderBlur = createShader("blur.vert", GL_VERTEX_SHADER, __LINE__);
	fragmentShaderBlur = createShader("blur.frag", GL_FRAGMENT_SHADER, __LINE__);

	shaderProgramBlur = createProgram(vertexShaderBlur, fragmentShaderBlur, __LINE__);

	// TODO: Use program.
	glUseProgram(shaderProgramBlur);

	// Eingabe in diesen Shader ist die Textur, in die die Szene gerendert wird.
	// An dieser Stelle wird die uniform Location für die Textur-Variable im Shader geholt.
	setUniformInt(shaderProgramBlur, "texture", 0);

	setUniformInt(shaderProgramBlur, "filterWidth", 10);
}


int initFBOTextures()
{
	// Textur (fuer Teapot Bild) anlegen
	glGenTextures (1, &teapotTextureId);
	glBindTexture (GL_TEXTURE_2D, teapotTextureId);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// Depth Buffer Textur anlegen
	glGenTextures (1, &depthTextureId);
	glBindTexture (GL_TEXTURE_2D, depthTextureId);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// FBO (fuer Teapot Textur) anlegen und Texturen zuweisen
	glGenFramebuffers (1, &teapotFB);
	glBindFramebuffer (GL_FRAMEBUFFER, teapotFB);
	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, teapotTextureId, 0);
	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTextureId, 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture (GL_TEXTURE_2D, teapotTextureId); // texture 0 is the teapot color buffer

	// check framebuffer status
	GLenum status = glCheckFramebufferStatus (GL_FRAMEBUFFER);
	switch (status)
	{
	case GL_FRAMEBUFFER_COMPLETE:
		cout << "FBO complete" << endl;
		break;
	case GL_FRAMEBUFFER_UNSUPPORTED:
		cout << "FBO configuration unsupported" << endl;
		return 1;
	default:
		cout << "FBO programmer error" << endl;
		return 1;
	}
	glBindFramebufferEXT (GL_FRAMEBUFFER, 0);
	return 0;
}

void keyboard(unsigned char key, int x, int y)
{
	// set parameters
	switch (key)
	{
		case 'b':
			useBlur = !useBlur;
			break;
	}
}

// Bildschirmfuellendes Rechteck zeichnen -> Fragment Program wird fuer jedes Pixel aufgerufen
void drawScreenFillingQuad()
{
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	glBegin(GL_QUADS);
	{
		glTexCoord2f(0,0);
		glVertex2f(-1,-1);
		glTexCoord2f(1,0);
		glVertex2f( 1,-1);
		glTexCoord2f(1,1);
		glVertex2f(1,1);
		glTexCoord2f(0,1);
		glVertex2f( -1,1);
	}
	glEnd();

	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
}

void drawSzene()
{
	// Clear window
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLoadIdentity();
	gluLookAt(10, 7, 10, 0, 0, 0, 0, 1, 0);

	glRotatef(alpha, 0, 1, 0);
	glutSolidTeapot(3);
}

void display()
{
	// Pumping Shader anschalten falls aktiviert
	glUseProgram( 0 );

	// falls Blur Shader aktiviert ist, muss in eine Textur gerendert werden
	if (useBlur){
		// 1 draw szene into FBO teapotFB
		glBindFramebuffer (GL_FRAMEBUFFER, teapotFB);      // activate fbo
		drawSzene();
		// FBO abschalten: jetzt wird wieder in den Framebuffer gerendert
		glBindFramebuffer (GL_FRAMEBUFFER, 0);      // deactivate fbo


		// 2 draw content of FBO teapotFB onto screen using blur
		//   fragment programm
		// Blur Shader aktivieren und bildschirmfuellendes Rechteck zeichnen

		glUseProgram( shaderProgramBlur ); // activate horizontal blur shader
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		drawScreenFillingQuad();
		glUseProgram( 0 );

		// TODO: Teapot oben drüber zeichnen.
	}
	else{
		drawSzene();
	}

	// Increment rotation angle
	alpha += 1;

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

/// Init glew so that the GLSL functionality will be available
void initGLEW()
{
	if(glewInit() != GLEW_OK){
		cout << "GLEW init failed!" << endl;
	}
}

void initGLUT(char** argv, int argc)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutCreateWindow("Glowing Teapot");
}

void registerCallbackFunctions()
{
	glutKeyboardFunc(keyboard);
	glutDisplayFunc(display);
	glutTimerFunc(25, timer, 0);	// Call timer() in 25 milliseconds
}

int main(int argc, char** argv)
{
	initGLUT(argv, argc);
	initGLEW();
	// OpenGL/GLSL initializations
	initGL();
	initFBOTextures();
	initGLSL();

	registerCallbackFunctions();

	// Enter main loop
	glutMainLoop();

	return 0;
}
