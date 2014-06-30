
#include <stdio.h>
#include "cuda.h"
#include "common.h"
#include <GL/glut.h>

#define N 512

GLfloat viewPosition[4] = {0.0, 5.0, 10.0, 1.0};  
GLfloat viewDirection[4] = {-0.0, -5.0, -10.0, 0.0};  
GLfloat viewAngle = 45.0;
GLfloat viewNear = 4.5;
GLfloat viewFar = 25.0;

GLfloat xRotationAngle = 0.0f;
GLfloat yRotationAngle = 0.0f;

GLfloat xRotationSpeed = 3.0f;
GLfloat yRotationSpeed = 4.5f;

GLfloat depthPixels[N*N];
GLfloat colorPixels[N*N];
GLfloat filteredPixels[N*N];

float focusDepth = 0.5f;
float sizeScale = 20.0f;

float *devColorPixelsSrc, *devColorPixelsDst, *devDepthPixels, *devSAT;

void drawGround()
{
	GLfloat grey[3] = {0.8,0.8,0.8};
    
	glNormal3f(0, 1, 0);
	glMaterialfv(GL_FRONT, GL_AMBIENT, grey);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, grey);
    glBegin(GL_QUADS);
    glVertex3f(-10, 0, 10);
    glVertex3f( 10, 0, 10);
    glVertex3f( 10, 0, -10);
    glVertex3f(-10, 0, -10);
    glEnd();
}


void drawScene()
{
    GLfloat diffuse1[4] = {0.5, 0.5, 0.5, 1.0};
	GLfloat lightAmbient[4] = {0.0, 0.0 ,0.0, 1.0};  
	GLfloat lightDiffuse[4] = {0.2, 0.2 ,0.2, 1.0};  
	GLfloat lightPosition[4] = {0.5, 10.5, 6.0, 1.0};  

	glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);
	glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, diffuse1);    
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse1);    
    glPushMatrix();
    glTranslatef(0.0, 1.0, 0.0);
	glRotatef(-yRotationAngle/3.0, 0.0f, 1.0f, 0.0f);
    glutSolidTeapot(1.0);
    glPopMatrix();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, diffuse1);    
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse1);    
    glPushMatrix();
    glTranslatef(-1.0, 1.0, 3.0);
	glRotatef(-yRotationAngle/3.0, 0.0f, 1.0f, 0.0f);
    glutSolidTeapot(1.0);
    glPopMatrix();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, diffuse1);    
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse1);    
    glPushMatrix();
    glTranslatef(1.0, 1.0, -3.0);
	glRotatef(-yRotationAngle/3.0, 0.0f, 1.0f, 0.0f);
    glutSolidTeapot(1.0);
    glPopMatrix();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, diffuse1);    
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse1);    
    glPushMatrix();
    glTranslatef(-2.0, 1.0, 6.0);
	glRotatef(-yRotationAngle/3.0, 0.0f, 1.0f, 0.0f);
    glutSolidTeapot(1.0);
    glPopMatrix();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, diffuse1);    
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse1);    
    glPushMatrix();
    glTranslatef(2.0, 1.0, -6.0);
	glRotatef(-yRotationAngle/3.0, 0.0f, 1.0f, 0.0f);
    glutSolidTeapot(1.0);
    glPopMatrix();

	drawGround();
}



void initGL()
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);               // OpenGL Lichtquellen aktivieren
	glEnable(GL_LIGHT0);                 // Lichtquelle 0 anmachen 

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    gluPerspective(viewAngle, 1.0f, viewNear, viewFar);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();   
}

__device__ void clipFilterMask(int x, int y, int filterSize, int imageWidth, int imageHeight, int clipped[4])
{
	int top = y + filterSize;
	int bottom = y - filterSize;
	int left = x - filterSize;
	int right = x + filterSize;

	top = (top >= imageHeight) ? imageHeight-1 : top;
	bottom = (bottom < 0) ? 0 : bottom;
	left = (left < 0) ? 0 : left;
	right = (right >= imageWidth) ? imageWidth-1 : right;

	clipped[0] = top;
	clipped[1] = bottom;
	clipped[2] = left;
	clipped[3] = right;
}

// image is a quadratic gayscaleImage 
// Pxy lies in image[y*SIZE+x]
__global__ void transposeQuadraticImage(float* image, float* image2, const int SIZE)
{
	int posX = blockIdx.x * blockDim.x + threadIdx.x;
	int posY = blockIdx.y * blockDim.y + threadIdx.y;

	int myIndex = posX*SIZE+posY;
	int transposeIndex = posY*SIZE+posX;
	float transPoseIntensity = image[transposeIndex];
	//__syncthreads();

	image2[myIndex] = transPoseIntensity;
}

// assumes one block = one row
__global__ void sat_filter(float *dstImage, float *sat, float *srcDepth, 
					   float focusDepth, float sizeScale, int n)
{
	int posY = blockIdx.x;
	int posX = threadIdx.x;
	int myIndex = posY * blockDim.x + posX;
	// TODO: Filtergröße bestimmen
	float filterSize = 1 + sizeScale * abs(srcDepth[myIndex]-focusDepth);
	// TODO: SAT-Werte für die Eckpunkte des Filterkerns bestimmen.
	int filterBorders[4];
	clipFilterMask(posX, posY, filterSize, N,N,filterBorders);
	int top = filterBorders[0];
	int bottom = filterBorders[1];
	int left = filterBorders[2];
	int right = filterBorders[3];
	float satTL = sat[top * N + left];
	float satTR = sat[top * N + right];
	float satBL = sat[bottom * N + left];
	float satBR = sat[bottom * N + right];
	// TODO: Anzahl der Pixel im Filterkern bestimmen	
	float filterNumPixels = (top - bottom) * (right - left);
	// TODO: Mittelwert berechnen.
	dstImage[myIndex] = (satTR - satTL - satBR + satBL)/filterNumPixels;
}


__global__ void scan_naive(float *g_odata, float *g_idata, int n)
{
    // Dynamically allocated shared memory for scan kernels
    __shared__  float temp[2*N];

    int thid = threadIdx.x;
    int bid = blockIdx.x;

    int pout = 0;
    int pin = 1;

    // Cache the computational window in shared memory
    temp[pout*n + thid] = (thid > 0) ? g_idata[bid * N + thid-1] : 0;

    for (int offset = 1; offset < n; offset *= 2)
    {
        pout = 1 - pout;
        pin  = 1 - pout;
        __syncthreads();

        temp[pout*n+thid] = temp[pin*n+thid];

        if (thid >= offset)
            temp[pout*n+thid] += temp[pin*n+thid - offset];
    }

    __syncthreads();

    g_odata[bid * N + thid] = temp[pout*n+thid];
}


void initCUDA()
{
    CUDA_SAFE_CALL(cudaMalloc( (void**)&devColorPixelsSrc, N * N * sizeof(float) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&devColorPixelsDst, N * N * sizeof(float) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&devDepthPixels, N * N * sizeof(float) ));
	CUDA_SAFE_CALL(cudaMalloc( (void**)&devSAT, N*N * sizeof(float) ));
}

void cleanCUDA()
{
	cudaFree( devColorPixelsSrc );
    cudaFree( devColorPixelsDst );
    cudaFree( devDepthPixels );
	cudaFree( devSAT );
}

void special(int key, int x, int y)
{
	switch (key) {
	case GLUT_KEY_UP :
		focusDepth += 0.05;
		 if (focusDepth > 1.0) focusDepth = 1.0;
		break;
	case GLUT_KEY_DOWN :
		focusDepth -= 0.05;
		 if (focusDepth < 0.0) focusDepth = 0.0;
		break;
	case GLUT_KEY_LEFT :
		sizeScale -= 1.0;
		if (sizeScale > 100.0) sizeScale = 100.0;
		break;
	case GLUT_KEY_RIGHT :
		sizeScale += 1.0;
		if (sizeScale < 1.0) sizeScale = 1.0;
		break;
	case GLUT_KEY_PAGE_UP :
		viewFar += 1.0;
		initGL();
		break;
	case GLUT_KEY_PAGE_DOWN :
		viewFar -= 1.0;
		if (viewFar < viewNear) viewFar = viewNear;
		initGL();
		break;
	}
}

__global__ void normalizeArray(float* p_array, int SIZE)
{
	int myIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if(p_array[SIZE] != 0)
	{
		p_array[myIndex] = p_array[myIndex]/p_array[SIZE]; 
	}
}

void display(void)								
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Szene rendern
	glLoadIdentity();
	gluLookAt(viewPosition[0], viewPosition[1], viewPosition[2], 
              viewDirection[0] - viewPosition[0], viewDirection[1] - viewPosition[1], viewDirection[2] - viewPosition[2], 
              0, 1, 0);    
	drawScene();	    

    // Tiefe und Farbe in den RAM streamen.
    glReadPixels( 0, 0, N, N, GL_DEPTH_COMPONENT, GL_FLOAT, depthPixels);	
    glReadPixels( 0, 0, N, N, GL_LUMINANCE, GL_FLOAT, colorPixels);	

    // Beide arrays in den Device-Memory kopieren.
    cudaMemcpy( devColorPixelsSrc, colorPixels, N * N * sizeof(GLfloat), cudaMemcpyHostToDevice );
    cudaMemcpy( devDepthPixels, depthPixels, N * N * sizeof(float), cudaMemcpyHostToDevice );

	dim3 gridSize(32,32);
	dim3 blockSize(16,16);

	// TODO: Scan    
	scan_naive<<<N,N>>>(devColorPixelsDst,devColorPixelsSrc,N);
	// TODO: Transponieren 
	transposeQuadraticImage<<<gridSize,blockSize>>>(devColorPixelsDst,devSAT,N);
	// TODO: Scan
	scan_naive<<<N,N>>>(devColorPixelsDst,devSAT,N);
	// TODO: Transponieren   
	transposeQuadraticImage<<<gridSize,blockSize>>>(devColorPixelsDst,devSAT,N);
	// TODO: SAT-Filter anwenden
	sat_filter<<<N,N>>>(devColorPixelsDst, devSAT, devDepthPixels, focusDepth, sizeScale, N);

	//normalizeArray<<<N,N>>>(devSAT,N*N);
	// Ergebnis in Host-Memory kopieren
	cudaMemcpy( filteredPixels, devColorPixelsDst, N*N*4, cudaMemcpyDeviceToHost );

	// TODO: Beim #if aus der 0 eine 1 machen, damit das gefilterte Bild angezeigt wird!
#if 1
	/*if(filteredPixels[N*N-1] != 0)
	{
		for(int i = 0; i < N*N; i++)
		{
			filteredPixels[i] /= filteredPixels[N*N-1];
		}
	}*/
	
	
	// Mittelwert-Bild rendern
	glDrawPixels( N, N, GL_LUMINANCE, GL_FLOAT, filteredPixels );
#else
	// Durchreichen des Eingabebildes.
	glDrawPixels( N, N, GL_LUMINANCE, GL_FLOAT, colorPixels );
#endif

	xRotationAngle += xRotationSpeed;   // Rotationswinkel erhoehen
	yRotationAngle += yRotationSpeed;

	glutSwapBuffers();
}

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(N, N);
	glutCreateWindow("Simple CUDA SAT Depth of Field");

	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutSpecialFunc(special);

	initGL();
	initCUDA();

	glutMainLoop();

	cleanCUDA();

	return 0;
}

