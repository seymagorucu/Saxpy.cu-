#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (id < n) // Prevents more than N operations
	{
		y[id] = a*x[id] + y[id]; 
       // printf( " y[id] %f , " , y[id] );
	}
}

void random_float(float* random, int size)
{
	for (int i=0;i<size;i++) 
	{
		random[i]=((float)rand()/(float)(RAND_MAX));
	}
}

int main(void)
{
	int N;
	float A;
	int nDevices;
	int max_threads_per_blok = 0;
	int max_grid_size = 0;
	int max_thread_blocks = 0;
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	cudaGetDeviceCount(&nDevices);
	printf("cudaGetDeviceCount: %d\n", nDevices);
	printf("There are %d CUDA devices.\n", nDevices);

	for (int i = 0; i < nDevices; i++) 
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d \n", i);
		printf("Device name: %s \n ", prop.name);
		printf("Block dimensions: %d x %d  x %d \n", prop.maxThreadsDim[0],prop.maxThreadsDim[1],  prop.maxThreadsDim[2]);
		printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
		max_threads_per_blok= prop.maxThreadsPerBlock;
		printf ("Grid dimensions:  %d x %d x %d \n", prop.maxGridSize[0],  prop.maxGridSize[1],  prop.maxGridSize[2]);
		max_grid_size = prop.maxGridSize[0];

		if (max_grid_size < prop.maxGridSize[1])
		{
			max_grid_size =  prop.maxGridSize[1];
		}
		else if (max_grid_size < prop.maxGridSize[2]) 
		{
			max_grid_size = prop.maxGridSize[2];
		}
		max_thread_blocks = max_grid_size / max_threads_per_blok; // prop.maxGridSize[0] / prop.maxThreadsDim[0] for this operation used x dimension
		printf (" Maximum number of thread blocks for x  = %d \n", max_thread_blocks);
	}

	printf("Please input an N value: ");
	scanf("%d", &N);

	printf("Please input an A value: ");
	scanf("%f", &A);	

	float *h_x, *h_y, *d_x, *d_y;
	size_t size = N * sizeof(float);

	// Allocate the host input x
	h_x = (float *)malloc(size);

	// Allocate the host input y
	h_y = (float *)malloc(size);

	// Verify that allocations succeeded
	if (h_x == NULL || h_y == NULL)
	{
		fprintf(stderr, "Failed to allocate host x and y\n");
		exit(EXIT_FAILURE);
	}

    random_float(h_x, N);
    random_float(h_y, N);

	d_x = NULL;
	err = cudaMalloc((void **)&d_x, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device  x (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	d_y = NULL;
	err = cudaMalloc((void **)&d_y, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device  y (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy  x from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	err = cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy  y from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	int number_of_blocks = (N+1023)/1024;

	printf("Print max_threads %d \n  " , (max_thread_blocks * max_threads_per_blok) );

	if( N <= (max_thread_blocks * max_threads_per_blok)) // cannot be greater than the total number of threads
	{        
		int number_of_threads_per_block = (N/number_of_blocks);
        //This control is added to avoid missing the number of threads when integer does not give value when number is divided.
		if (N % number_of_blocks != 0 && number_of_threads_per_block < 1024)
		{
			 number_of_threads_per_block = number_of_threads_per_block+1;
		}

		if (number_of_blocks <= max_thread_blocks  )
		{
			printf (" saxpy <<<number_of_blocks = %d , number_of_threads_per_block = %d >>>\n ",number_of_blocks ,number_of_threads_per_block);
			saxpy<<<number_of_blocks ,number_of_threads_per_block >>>(N, A, d_x, d_y);

			err = cudaGetLastError();

			if (err != cudaSuccess)
			{
				fprintf(stderr, "Failed to launch saxpy kernel (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
		   
		}  
	}
    else
    {
        printf ("N number is too large, please enter a smaller number\n");
    }

	err = cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy  y from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
  
	err = cudaFree(d_x);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device  x (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_y);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device  y (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	free(h_x);
	free(h_y);
}
