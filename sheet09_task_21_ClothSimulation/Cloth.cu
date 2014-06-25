
// Includes
#include "CudaMath.h"
#include "Cloth.h"

// Computes the impacts between two points that are connected by a constraint in order to satisfy the constraint a little better.
__device__ float3 computeImpact(float3 me, float3 other, float stepsize, float h)
{	
	const float aimedDistance = 1.0 / (float)RESOLUTION_X;
	float3 dir = other-me;
	float ldir = length(dir);
	if (ldir==0) return dir;
	float e = (ldir - aimedDistance) * 0.5;
	float3 debug = dir/ldir * e / (h*h) * stepsize; 
	return debug;
}

// Simple collision detection against a sphere at (0,0,0) with radius SPHERE_RADIUS and skin width SKIN_WIDTH.
__device__ float3 sphereCollision(float3 p, float h)
{
	// TODO: Testen, ob Punkt im inneren der Kugel ist. Wenn ja, dann einen Impuls berechnen, der sie wieder heraus bewegt.
	float centerDist = length(p);
	if(centerDist < SPHERE_RADIUS+SKIN_WIDTH)
	{
		return (p)/h;
	}
	else
	{
		return make_float3(0,0,0);
	}
	//return p;
}

// -----------------------------------------------------------------------------------------------
// Aufsummieren der Impulse, die von den benachbarten Gitterpunkten ausgeübt werden.
// impacts += ...
__global__ void computeImpacts(float3* oldPos, float3* impacts, float stepsize, float h)
{
	// TODO: Positionen der benachbarten Gitterpunkte und des eigenen Gitterpunktes ablesen.
	int myIndex = blockIdx.x * RESOLUTION_Y + blockIdx.y;
	int up = blockIdx.y -1;
	int down = blockIdx.y +1;
	int left = blockIdx.x -1;
	int right = blockIdx.x +1;

	float3 impactsTmp = make_float3(0, 0, 0);

	// TODO: Kollisionsbehandlung mit Kugel durchführen.
	impacts[myIndex] += sphereCollision(oldPos[myIndex],h);

	// TODO: Mit jedem Nachbar besteht ein Constraint. Dementsprechend für jeden Nachbar 
	//		 computeImpact aufrufen und die Ergebnisse aufsummieren.
	if(up > 0)
	{
		int upIdx = blockIdx.x * RESOLUTION_Y + up;
		float3 tmp = computeImpact(oldPos[myIndex], oldPos[upIdx],stepsize,h);
		impactsTmp = impactsTmp + tmp;
	}
	if(down < RESOLUTION_Y)
	{
		int downIdx = blockIdx.x * RESOLUTION_Y + down;
		float3 tmp = computeImpact(oldPos[myIndex], oldPos[downIdx],stepsize,h);
		impactsTmp = impactsTmp + tmp;
	}
	if(left > 0)
	{
		int leftIdx = left * RESOLUTION_Y + blockIdx.y;
		float3 tmp = computeImpact(oldPos[myIndex], oldPos[leftIdx],stepsize,h);
		impactsTmp = impactsTmp + tmp; 
	}
	if(right < RESOLUTION_X)
	{
		int rightIdx = right * RESOLUTION_Y + blockIdx.y;
		float3 tmp = computeImpact(oldPos[myIndex], oldPos[rightIdx],stepsize,h);
		impactsTmp = impactsTmp + tmp;
	}

	// TODO: Die Summe der Impulse auf "impacts" des eigenen Gitterpunkts addieren.	
	bool debug = !(impactsTmp.x == 0 && impactsTmp.y == 0 && impactsTmp.z == 0);
	if(debug)
	{
		impacts[myIndex] = impacts[myIndex] + impactsTmp;
	}
}

// -----------------------------------------------------------------------------------------------
// Preview-Step
// newpos = oldpos + (velocity + impacts * h) * h
__global__ void previewSteps(	float3* newPos, float3* oldPos, float3* impacts, float3* velocity,								
								float h)
{
	int index = blockIdx.x * RESOLUTION_Y + blockIdx.y;
	// TODO: Berechnen, wo wir wären, wenn wir eine Integration von der bisherigen Position 
	//		 mit der bisherigen Geschwindigkeit und den neuen Impulsen durchführen.
	newPos[index] = oldPos[index] + velocity[index] * h + impacts[index] * h*h;
	
	//newPos[index] = oldPos[index] - make_float3(0,0.002,0);
}

// -----------------------------------------------------------------------------------------------
// Integrate velocity
// velocity = velocity * LINEAR_DAMPING + (impacts - (0,GRAVITY,0)) * h 
__global__ void integrateVelocity(	float3* impacts, float3* velocity, float h)
{
	int index = blockIdx.x * RESOLUTION_Y + blockIdx.y;
	// TODO: Update velocity.
	float3 debug = velocity[index] * LINEAR_DAMPING + (impacts[index] - make_float3(0,GRAVITY,0)) * h; 
	velocity[index] = debug;
}

// -----------------------------------------------------------------------------------------------
// Test-Funktion die nur dazu da ist, damit man etwas sieht, sobald die VBOs gemapped werden...
__global__ void test( float3* newPos, float3* oldPos, float h)
{
	newPos[blockIdx.x] = oldPos[blockIdx.x] + make_float3(0, -h, 0);
}

__global__ void computeNormals(float3* pos, float3* normals)
{
	int myIndex = blockIdx.x * RESOLUTION_Y + blockIdx.y;
	int up = blockIdx.y -1;
	int down = blockIdx.y +1;
	int left = blockIdx.x -1;
	int right = blockIdx.x +1;

	float3 upv = make_float3(0,0,0);
	float3 downv = make_float3(0,0,0);
	float3 leftv = make_float3(0,0,0);
	float3 rightv = make_float3(0,0,0);

	if(up < 0)
	{
		upv = -1 * pos[down]-pos[myIndex];
	}
	else
	{
		upv = pos[up]-pos[myIndex];
	}
	if(down >= RESOLUTION_Y)
	{
		downv = -1 * pos[up]-pos[myIndex];
	}
	else
	{
		downv = pos[down]-pos[myIndex];
	}
	if(left < 0)
	{
		leftv = -1 * pos[right]-pos[myIndex];
	}
	else
	{
		leftv = pos[left]-pos[myIndex];
	}
	if(right >= RESOLUTION_X)
	{
		rightv = -1 * pos[left] - pos[myIndex];
	}
	else
	{
		rightv = pos[right]-pos[myIndex];
	}

	float3 n1 = cross(upv, rightv);
	float3 n2 = cross(downv, leftv);

	normals[myIndex] = (n1+n2)/2;

}

// -----------------------------------------------------------------------------------------------
void updateCloth(	float3* newPos, float3* oldPos, float3* impacts, float3* velocity, float3* normals,					
					float h, float stepsize)
{
	// dont move the row resY
	dim3 blocks(RESOLUTION_X, RESOLUTION_Y-1, 1);
	dim3 blockSIze(1,1);

	// -----------------------------
	// Clear impacts
	cudaMemset(impacts, 0, RESOLUTION_X*RESOLUTION_Y*sizeof(float3));

	// Updating constraints is an iterative process.
	// The more iterations we apply, the stiffer the cloth become.
	for (int i=0; i<10; ++i)
	{
		// -----------------------------
		// TODO: previewSteps Kernel aufrufen (Vorhersagen, wo die Gitterpunkte mit den aktuellen Impulsen landen würden.)
		// newpos = oldpos + (velocity + impacts * h) * h		
		previewSteps<<<blocks,1>>>(newPos, oldPos,impacts,velocity,h);
		// -----------------------------
		// TODO: computeImpacts Kernel aufrufen (Die Impulse neu berechnen, sodass die Constraints besser eingehalten werden.)
		// impacts += ...
		computeImpacts<<<blocks,1>>>(newPos,impacts,stepsize,h);
	}

	// -----------------------------
	// TODO: Approximieren der Normalen
	computeNormals<<<blocks,1>>>(newPos, normals);

	// -----------------------------
	// TODO: Integrate velocity kernel ausführen
	// Der kernel berechnet:  velocity = velocity * LINEAR_DAMPING + (impacts - (0,GRAVITY,0)) * h 	
	integrateVelocity<<<blocks,1>>>(impacts,velocity,h);
	//previewSteps<<<blocks,1>>>(newPos, oldPos,impacts,velocity,h);
}