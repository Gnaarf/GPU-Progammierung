
#include <stdio.h>
#include <math.h>
#include "common.h"
#include "bmp.h"
#include <stdlib.h>
#include <GL/freeglut.h>

#define DIM 512
#define blockSize 8

#define PI 3.1415926535897932f
#define centerX (DIM/2)
#define centerY (DIM/2)

float sourceColors[DIM*DIM];	// host memory for source image
float readBackPixels[DIM*DIM];	// host memory for swirled image

float *sourceDevPtr;			// device memory for source image
float *swirlDevPtr;				// device memory for swirled image

const int PIC_SIZE = DIM * DIM;
const int PIC_BYTE_SIZE = PIC_SIZE * sizeof(float);

float a = 0;
float b = 0;

__device__ void calcRotatedPos(float alpha, float2 inPos, int2& calcPos)
{
	inPos.x -= centerX;
	inPos.y -= centerY;
	float cosA = cos(alpha);
	float sinA = sin(alpha);
	float resx = cosA * inPos.x - sinA * inPos.y;
	float resy = sinA * inPos.x + cosA * inPos.y;
	//calcPos.x = cosA * inPos.x - sinA * inPos.y;
	//calcPos.y = sinA * inPos.x - cosA * inPos.y;
	resx += centerX;
	resy += centerY;
	calcPos.x = roundf(resx);
	calcPos.y = roundf(resy);
}

__global__ void swirlKernel( float *sourcePtr, float *targetPtr, float a, float b ) 
{
	int index1 = 0;
    // TODO: Index berechnen	
	/*int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
	int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
	index = pixelX + pixelY * blockDim.x * gridDim.x;*/

	// von Cord
	int blockSz = blockDim.x * blockDim.y;
	int i = threadIdx.x + threadIdx.y * blockDim.x + blockSz * (blockIdx.x + blockIdx.y * gridDim.x); 
	index1 = i;

	
	float2 pix;
	pix.x = i%DIM;
	pix.y = i/DIM;

	float2 pic_center;
	pic_center.x = centerX;
	pic_center.y = centerY;

	float2 radv;
	radv.x = abs(pix.x - pic_center.x);
	radv.y = abs(pix.y - pic_center.y);

	float r = sqrt(radv.x * radv.x + radv.y * radv.y);

	float alpha = a * pow(r,b);

	// TODO: Den swirl invertieren.
	int2 grabPos;
	calcRotatedPos(alpha, pix, grabPos);

	if(grabPos.x >= DIM || grabPos.x < 0)
	{
		grabPos.x = pix.x;
		grabPos.y = pix.y;
	}
	else if(grabPos.y >= DIM || grabPos.y < 0)
	{
		grabPos.x = pix.x;
		grabPos.y = pix.y;
	}
	int index2 = 0;
	index2 = grabPos.x + grabPos.y * DIM;

	targetPtr[index1] = sourcePtr[index2];    
}

void keyboard(unsigned char key, int x, int y)
{
	const float aStep = 0.01;
	const float bStep = 0.005;
	switch (key)
	{
		case 27:
			exit(0);	
			break;
		case '1': 
			a -= aStep; 
			a = std::max(a,-2.0f);

			std::cout << "a: " << a << std::endl;
			break;
		case '2': 
			a += aStep;
			a = std::min(a, 2.0f);

			std::cout << "a: " << a << std::endl;
			break;

		case '3': 
			b -= bStep;
			b = std::max(b, 0.0f);

			std::cout << "b: " << b << std::endl;
			break;
		case '4': 
			b += bStep;
			b = std::min(b, 1.0f);

			std::cout << "b: " << b << std::endl;
			break;
		case '+':
			
			break;
		case '-':
			
			break;
		case 'w':
			
			break;
	}
	glutPostRedisplay();
}

void display(void)	
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// TODO: Swirl Kernel aufrufen.
	dim3 gridSize(64,64);
	dim3 blocksize(8,8);
	swirlKernel<<<1024 ,256>>>(sourceDevPtr, swirlDevPtr, a,b);

	// TODO: Ergebnis zu host memory zuruecklesen.
	cudaMemcpy(readBackPixels, swirlDevPtr,PIC_BYTE_SIZE, cudaMemcpyDeviceToHost);

	// Ergebnis zeichnen (ja, jetzt gehts direkt wieder zur GPU zurueck...) 
	glDrawPixels( DIM, DIM, GL_LUMINANCE, GL_FLOAT, readBackPixels );

	glutSwapBuffers();
}

// clean up memory allocated on the GPU
void cleanup() {
    CUDA_SAFE_CALL( cudaFree( sourceDevPtr ) ); 
    CUDA_SAFE_CALL( cudaFree( swirlDevPtr ) ); 
}

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("Simple OpenGL CUDA");
	glutIdleFunc(display);
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);

	// load bitmap	
	Bitmap bmp = Bitmap("who-is-that.bmp");
	if (bmp.isValid())
	{		
		for (int i = 0 ; i < DIM*DIM ; i++) {
			sourceColors[i] = bmp.getR(i/DIM, i%DIM) / 255.0f;
		}
	}

	// TODO: allocate memory at sourceDevPtr on the GPU and copy sourceColors into it.
	CUDA_SAFE_CALL( cudaMalloc((void**)&sourceDevPtr, PIC_BYTE_SIZE) );
	CUDA_SAFE_CALL( cudaMemcpy(sourceDevPtr, sourceColors, PIC_BYTE_SIZE, cudaMemcpyHostToDevice) );
	
	// TODO: allocate memory at swirlDevPtr for the unswirled image.	
	CUDA_SAFE_CALL( cudaMalloc((void**)&swirlDevPtr, PIC_BYTE_SIZE) );

	glutMainLoop();

	cleanup();
}
