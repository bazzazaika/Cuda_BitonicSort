#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <intrin.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand_kernel.h>

#include <algorithm>

//#define SIZE 536870912
int SIZE;
//2^12 4096
//2^20 1048576
//2^25 33554432
// 2^26 67108864
// 2^27 134217728
// 2^28 268435456

void generate(int* arr, int length)
{

	srand(time(NULL));
	unsigned int i;
	for (i = 0; i < length; ++i)
	{
		arr[i] = rand();
	}
}

__device__ void swap(int* one, int* two)
{
	int temp = *one;
	*one = *two;
	*two = temp;
}

__global__ void Ker_bitonicSort(int* mas, int k, int j, int SIZE)
{
	unsigned int i;
	i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < SIZE)
	{
		int ind = i | j;//узнаем индекс элемента, с которым хотим сравнивать
		if ((i & k) == 0)
		{
			if (mas[i] > mas[ind])
			{
				swap(&mas[i], &mas[ind]);
			}
		}
		else
		{
			if (mas[i] < mas[ind])
			{
				swap(&mas[i], &mas[ind]);
			}
		}
	}
}


void printArr(int* dev_values, int j, int k)
{
	printf("j = %d, k = %d\n", j, k);
	for (int i = 0; i < SIZE; i++)
	{
		printf("%d ", dev_values[i]);
	}
	printf("\n\n");
}

void bitonicSort(int* mas)
{
	int* cuda_mas;
	size_t size = SIZE * sizeof(int);
	cudaMalloc((void**)&cuda_mas, size);
	cudaMemcpy(cuda_mas, mas, size, cudaMemcpyHostToDevice);
	int threads = 1024;
	int blocks = (SIZE / threads == 0) ? 1 : SIZE / threads;
	int j, k;

	for (k = 2; k <= SIZE; k <<= 1)
	{
		for (j = k >> 1; j > 0; j = j >> 1)
		{
			Ker_bitonicSort << <blocks, threads >> > (cuda_mas, k, j, SIZE);
			//cudaMemcpy(mas, cuda_mas, size, cudaMemcpyDeviceToHost);
			//printArr(mas, j, k);
		}
	}
	cudaMemcpy(mas, cuda_mas, size, cudaMemcpyDeviceToHost);
	cudaFree(cuda_mas);
	//printf("Done on GPU!\n");
}

int check(int* values)
{
	int i;
	for (i = 0; i < SIZE - 1;i++)
	{
		if (values[i] > values[i + 1])
			return 0;
	}
	//printf("Good!\n");
	return 1;
}
int main(void)
{
	unsigned long long start, stop;

	//int mas[8] = { 4,3,9,5,6,2,1,7 };
	//printArr(mas, 0, 0);
	SIZE = 268435456;//268435456
	printf("SIZE = %d\n", SIZE);
	int* mas = (int*)malloc(SIZE * sizeof(int));
	generate(mas, SIZE);
	//printArr(mas, 0, 0);
	//start = __rdtsc();
	bitonicSort(mas);
	//stop = __rdtsc();
	if (check(mas))
		printf("Good!\n");
		//printf("TIME = %llu\n", (stop - start));


}