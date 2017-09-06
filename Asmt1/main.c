#include <time.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int c_conv(int in_channel, int o_channel, int kernel_size, int stride, double **R, double **G, double **B, double **kernel)
{
	int x_new = floor(((720 - kernel_size) / stride) + 1);
	int y_new = floor(((1280 - kernel_size) / stride) + 1);
	int num_ops = 0;

	double resultR = 0, resultG = 0, resultB = 0, result = 0;

	double ***output_image = malloc(sizeof(double**)*x_new);
	for (int j = 0; j < x_new; j++)
	{
		output_image[j] = (double**)malloc(y_new*sizeof(double*));
		for (int k = 0; k < y_new; k++)
		{
			output_image[j][k] = (double*)malloc(o_channel*sizeof(double));

		}
	}
	for (int ch = 0; ch < o_channel; ch++)
	{
		for (int i = 0; i < x_new; i++)
		{
			for (int j = 0; j < y_new; j++)
			{
				for (int n1 = 0; n1 < kernel_size; n1++)
				{
					for (int n2 = 0; n2 < kernel_size; n2++)
					{
						resultR += R[i*stride + n1][j*stride + n2] * kernel[n1][n2];
						resultG += G[i*stride + n1][j*stride + n2] * kernel[n1][n2];
						resultB += B[i*stride + n1][j*stride + n2] * kernel[n1][n2];
					}
				}
				result = resultR + resultG + resultB;
				output_image[i][j][ch] = result;
				num_ops = num_ops + (kernel_size*kernel_size*in_channel) + (in_channel*(kernel_size*kernel_size - 1)) + in_channel - 1;
			}
		}
	}
	return num_ops;
}

int main(int argc, char *argv[])
{
	clock_t start, end;
	double time_taken;

	int n_rows = 720, n_cols = 1280;

	double **R = malloc(sizeof(double*)*n_rows);
	double **G = malloc(sizeof(double*)*n_rows);
	double **B = malloc(sizeof(double*)*n_rows);
	double **kernel = malloc(sizeof(double*)*3);
	for (int j = 0; j < n_rows; j++)
	{
		R[j] = (double*)malloc(n_cols*sizeof(double));
		G[j] = (double*)malloc(n_cols*sizeof(double));
		B[j] = (double*)malloc(n_cols*sizeof(double));
	}
	for (int j = 0; j < 3; j++)
	{
		kernel[j] = (double*)malloc(3*sizeof(double));
	}
	for (int i = 0; i < n_rows; i++)
	{
		for (int j = 0; j < n_cols; j++)
		{
			R[i][j] = rand();
			G[i][j] = rand();
			B[i][j] = rand();
		}
	}

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			kernel[i][j] = rand();
		}
	}
	
	for (int i = 0; i <= 10; i++)
	{
		start = clock();
		c_conv(3, pow(2, i), 3, 1, R, G, B, kernel);
		end = clock();
		time_taken = (double)(end - start) / CLOCKS_PER_SEC;
		printf("o_channel %d = %.6f secs \n", (int)pow(2, i), time_taken);
	}
	return 0;
}