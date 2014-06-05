
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

__device__ void calcRotatedPos(float alpha, float2 inPos, float2& calcPos)
{
	inPos.x -= centerX;
	inPos.y -= centerY;
	float cosA = cos(alpha);
	float sinA = sin(alpha);
	float resx = cosA * inPos.x - sinA * inPos.y;
	float resy = sinA * inPos.x - cosA * inPos.y;
	//calcPos.x = cosA * inPos.x - sinA * inPos.y;
	//calcPos.y = sinA * inPos.x - cosA * inPos.y;
	resx += centerX;
	resy += centerY;
	calcPos.x = resx;
	calcPos.y = resy;
}

__global__ void swirlKernel( float *sourcePtr, float *targetPtr, float a, float b ) 
{
	int index = 0;
    // TODO: Index berechnen	
	/*int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
	int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
	index = pixelX + pixelY * blockDim.x * gridDim.x;*/

	// von Cord
	int blockSz = blockDim.x * blockDim.y;
	int i = threadIdx.x + threadIdx.y * blockDim.x + blockSz * (blockIdx.x + blockIdx.y * gridDim.x); 
	index = i;

	
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
	float2 grabPos;
	calcRotatedPos(alpha, pix, grabPos);

	index = grabPos.x + grabPos.y * DIM;

	targetPtr[index] = sourcePtr[index];    // simple copy
}

void display(void)	
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// TODO: Swirl Kernel aufrufen.
	dim3 gridSize(64,64);
	dim3 blocksize(8,8);
	swirlKernel<<<1024 ,256>>>(sourceDevPtr, swirlDevPtr, 1,0.3);

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
