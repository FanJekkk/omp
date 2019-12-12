#include <iostream>
#include <omp.h> //OpenMP
#include <condition_variable>
#define A 110351515245
#define B 12345
#ifndef __cplusplus
#include <stdalign.h>
#endif // __cplusplus
#include <iomanip>
#define CACHE_SIZE 64
#define CACHE_SIZE_(x) (CACHE_SIZE / sizeof(x))
#define N 10000000
typedef struct aligned_accumulator
{
	alignas(64) unsigned value;
} alligned_element;

double median_reduction(alligned_element* pArray, unsigned count, unsigned min_val, unsigned max_val);
double median_reduction_v2(int* pArray, unsigned count, unsigned min_val, unsigned max_val);
double median_reduction_v3(int* pArray, unsigned count, unsigned min_val, unsigned max_val);

unsigned lin_con(unsigned x);

int main()
{
	int MaxT = omp_get_num_procs();

	std::cout << "==OMP Reduction==";
	std::cout << "\n";
	alligned_element* pArray = (alligned_element*)malloc(N * sizeof(alligned_element));
	for (int i = 1; i <= MaxT; i++)
	{
		omp_set_num_threads(i);
		auto t1 = omp_get_wtime();
		double res = median_reduction(pArray, N, 0, 1000);
		auto t2 = omp_get_wtime();
		std::cout << std::setprecision(5) << "MedianReduction(" << N << "," << i << ") = " << res << "   ";
		std::cout << std::setprecision(0) << std::fixed << (t2 - t1) * 1000 << " ms" << "\n";
		// std::cout << "    " << std::setprecision(5) << std::fixed << ((time_seq * 1000) / ((tm_omp2_padding - tm_omp1_padding) * 1000)) << std::endl;
	}
	free(pArray);

	std::cout << "\n";

	int* pArray_v2;
	pArray_v2 = (int*)malloc(N * sizeof(int));
	for (int i = 1; i <= MaxT; i++)
	{
		omp_set_num_threads(i);
		auto t1_v2 = omp_get_wtime();
		double res_v2 = median_reduction_v2(pArray_v2, N, 0, 1000);
		auto t2_v2 = omp_get_wtime();
		std::cout << std::setprecision(5) << "MedianReduction2(" << N << "," << i << ") = " << res_v2 << "   ";
		std::cout << std::setprecision(0) << std::fixed << (t2_v2 - t1_v2) * 1000 << " ms" << "\n";
	}
	free(pArray_v2);

	std::cout << "\n";

	int* pArray_v3;
	pArray_v3 = (int*)malloc(N * sizeof(int));
	for (int i = 1; i <= MaxT; i++)
	{
		omp_set_num_threads(i);
		auto t1_v3 = omp_get_wtime();
		double res_v3 = median_reduction_v3(pArray_v3, N, 0, 1000);
		auto t2_v3 = omp_get_wtime();
		std::cout << std::setprecision(5) << "MedianReduction3(" << N << "," << i << ") = " << res_v3 << "   ";
		std::cout << std::setprecision(0) << std::fixed << (t2_v3 - t1_v3) * 1000 << " ms" << "\n";
	}
	free(pArray_v3);

	std::cout << "\n";

	return 0;
}

double median_reduction(alligned_element* pArray, unsigned count, unsigned min_val, unsigned max_val)
{


	double eMedian = -1;
	int T;
	aligned_accumulator* pSums = (aligned_accumulator*)malloc((size_t)omp_get_num_procs() * sizeof(aligned_accumulator));

#pragma omp parallel shared(eMedian) private(T) \
    firstprivate(pArray, count, min_val, max_val)
	{
#pragma omp single copyprivate(T)
		{
			T = omp_get_num_threads(); // копируется значение T в приватную память потоков
		}

		int t = omp_get_thread_num(), i;
		int rnd_val = t + (int)omp_get_wtime();
		int iNeighbourOffset;

		for (i = t; (unsigned)i < count; i += T)
		{
			rnd_val = lin_con(rnd_val);
			pArray[i].value = (rnd_val % (max_val - min_val)) + min_val;
		}
#pragma omp barrier

#pragma omp single nowait
		{
			double sum = 0;
#pragma omp parallel for reduction(+:sum)
			for (i = 0; i < (int)count; ++i)
				sum += pArray[i].value;
			eMedian = (double)sum / count;

		}


	}
	return eMedian;
}

double median_reduction_v2(int* pArray, unsigned count, unsigned min_val, unsigned max_val)
{


	double eMedian = -1;
	int T;
	aligned_accumulator* pSums = (aligned_accumulator*)malloc((size_t)omp_get_num_procs() * sizeof(aligned_accumulator));

#pragma omp parallel shared(eMedian) private(T) \
    firstprivate(pArray, count, min_val, max_val)
	{
#pragma omp single copyprivate(T)
		{
			T = omp_get_num_threads(); // копируется значение T в приватную память потоков
		}

		int t = omp_get_thread_num(), i, step = T * CACHE_SIZE_(int);
		int rnd_val = t + (int)omp_get_wtime();
		for (i = t * CACHE_SIZE_(int); (unsigned)i < count; i += step)
		{
			int j, j_max = i + CACHE_SIZE_(int);
			if (count < (unsigned)j_max)
				j_max = (int)count;
			for (j = i; j < j_max; ++j)
			{
				rnd_val = lin_con(rnd_val);
				pArray[j] = (rnd_val % (max_val - min_val)) + min_val;
			}


		}
#pragma omp barrier

#pragma omp single nowait
		{
			long long int sum = 0;
#pragma omp parallel for reduction(+:sum)
			for (i = 0; i < (int)count; ++i)
				sum += pArray[i];
			eMedian = (double)sum / count;

		}
	}
	return eMedian;
}

double median_reduction_v3(int* pArray, unsigned count, unsigned min_val, unsigned max_val)
{

	double eMedian = -1;
	int T;
	aligned_accumulator* pSums = (aligned_accumulator*)malloc(omp_get_num_procs() * sizeof(aligned_accumulator));
#pragma omp parallel shared(pSums, eMedian) private(T) firstprivate(pArray_v3, pSums, count, min_val, max_val)
	{
#pragma omp single copyprivate(T)
	{
		T = omp_get_num_threads(); // копируется значение T в приватную память потоков
	}
	int t = omp_get_thread_num(), i, step = T * CACHE_SIZE_(int);
	int rnd_val = t + (int)omp_get_wtime();
	int iNeighbourOffset;
	long long sum = 0;
	for (int i = t * CACHE_SIZE_(int); (unsigned)i < count; i += step)
	{
		int j, jmax = i + CACHE_SIZE_(int);
		if ((unsigned) jmax > count)
			jmax = (int) count;
		for (j = i; j < jmax; ++j)
		{
			rnd_val = lin_con(rnd_val);
			sum += pArray[j] = rnd_val % (max_val - min_val) + min_val;
		}
	}
	pSums[t].value = sum;
	iNeighbourOffset = 1;
#pragma omp barrier
	if (count < T)
		T = count;
	while (iNeighbourOffset < T)
	{
		if (t + iNeighbourOffset < T && t % (2 * iNeighbourOffset) == 0)
			pSums[t].value += pSums[t + iNeighbourOffset].value;
#pragma omp barrier
		iNeighbourOffset *= 2;
	}
	if (t == 0)
		eMedian = (double)pSums[0].value / count;
	}
	free(pSums);
	return eMedian;
	}

	
unsigned lin_con(unsigned x)
{
	return x * A + B;
}
