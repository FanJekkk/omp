#include <vector>
#include <thread>
#include <omp.h>
#include <mutex>
#include <stdio.h>
#include <assert.h>
#include <iostream>

#define STEPS 100000000

union elem_t {
	char padding[64];
	double value;
};

double f(double x)
{
	return 4 / (1 + x * x);
}
double Integral_omp(double a, double b, double (*pFunc) (double))
{
	double dx = (b - a) / STEPS;
	double res = 0;
#pragma omp parallel
	{
		int T = omp_get_num_threads();
		int t = omp_get_thread_num();
		double r = 0;
		for (int i = t; i < STEPS; i += T)
			r += pFunc(dx * ((double)i + 0.5) + a);
#pragma omp critical
		{
			res += dx * r;
		}
	}

	return res;

}


double integral_seq(double a, double b)
{
	double dx = (b - a) / STEPS;
	double res = 0;

	for (int i = 0; i < STEPS; ++i)
		res += f((i + 0.5) * dx + a);
	res *= dx;

	return res;

}

double Integral_cpp(double a, double b, double (*pFunc) (double))
{
	auto T = std::thread::hardware_concurrency();
	double dx = (b - a) / STEPS;
	std::mutex mtx;

	std::vector<std::thread> threads;
	double res = 0;
	for (unsigned t = 0; t < T; ++t)
	{
		threads.emplace_back(std::thread([T, t, dx, &res, a, &mtx]()
		{
			double r = 0;
			for (auto i = t; i < STEPS; i += T)
				r += f(dx * ((double)i + 0.5) + a);
			mtx.lock();
			res += r;
			mtx.unlock();
		}));
	}
	for (auto& thr : threads)
		thr.join();
	return res * dx;
}

double Integral_razd(double a, double b, double(*pFunc)(double))
{
	double sum = 0;
	double dx = (b - a) / STEPS;
	double* pSumms;
	size_t g_cThreads = 0;
	size_t i;
	size_t cSumms = (size_t)omp_get_num_procs();
	assert(cSumms > 1);
	pSumms = (double*)calloc(cSumms, sizeof(double));
	assert(pSumms != NULL);
#pragma omp parallel
	{
		unsigned i;
		size_t iThread = (size_t)omp_get_thread_num(), cThreads = (size_t)omp_get_num_threads();
		if (iThread == 0)
		{
			g_cThreads = cThreads;
		}
		for (i = iThread; i < STEPS; i += cThreads)
			pSumms[iThread] += pFunc(a + dx * (i + 0.5));
	}
	for (i = 0; i < g_cThreads; ++i)
		sum += pSumms[i];
	sum *= dx;
	free(pSumms);
	return sum;
}

double Integral_padding(double a, double b, double(*pFunc)(double))
{
	double result = 0;
	unsigned P = omp_get_num_procs();
	double dx = (b - a) / STEPS;
	elem_t* pResults = (elem_t*)malloc(P * sizeof(elem_t));
	unsigned T;
#pragma omp parallel
	{
		unsigned t = omp_get_thread_num();
#pragma omp single
		{
			T = omp_get_num_threads();
		}
		pResults[t].value = 0;
		for (unsigned i = t; i < STEPS; i += T)
			pResults[t].value += f((i + 0.5) * dx + a);
	}
	for (unsigned i = 0; i < T; ++i)
		result += pResults[i].value;
	return result * dx;
}

double Integral_Reduction(double a, double b, double(*pFunc)(double))
{
	double dx = (b - a) / STEPS;
	double result = 0;
#pragma omp parallel for reduction(+:result)
		for (unsigned i = 0; i < STEPS; ++i)
			result += f(dx * (i + 0.5) + a);
	return result * dx;
}

struct run_result
{
	double resulting_value;
	double time_ms;
};

run_result
run_experiment_threads(std::size_t cThreadCount, double (*Integral) (double a, double b, double(*pFunc)(double)))
{
	long double t1;
	run_result result;
	omp_set_num_threads(int(cThreadCount));
	t1 = omp_get_wtime();
	result.resulting_value = Integral(0, 1, f);
	result.time_ms = (omp_get_wtime() - t1) / 1000;
	return result;
}

std::vector<run_result>
run_experiment(double (*Integral) (double a, double b, double(*pFunc) (double)))
{
	std::vector<run_result>
		results;
	auto cParallelism = std::size_t(omp_get_num_procs());
	results.reserve(cParallelism);
	for (std::size_t i = 0; i < cParallelism;)
	results.emplace_back(run_experiment_threads(++i, Integral));
	return results;
}

std::ostream& operator << (std::ostream & os, const run_result& val)
{
	return os << "Result:\t" << val.resulting_value << "\tTime:\t" << val.time_ms << "\tms.";
}

std::vector<run_result>
run_experiment(double (*Integral) (double a, double b, double (*pFunc)(double)));

std::ostream& operator<<(std::ostream& os, const run_result& val);

int main(int argc, char** argv)
{

	int MaxT = omp_get_num_procs();

	/*double t4 = omp_get_wtime();
	double result_cpp3 = integral_seq(0, 1);
	double t5 = omp_get_wtime();
	std::cout << "integral_seq(0,1) = " << result_cpp3 << std::endl;
	std::cout << (t5 - t4) * 1000 << " ms" << std::endl;
	std::cout << std::endl;

	double t8 = omp_get_wtime();
	double result_cpp2 = Integral_omp(0, 1, f);
	double t9 = omp_get_wtime();
	std::cout << "Integral_omp(0,1) = " << result_cpp2 << std::endl;
	std::cout << (t9 - t8) * 1000 << " ms" << std::endl;
	std::cout << std::endl;

	double t2 = omp_get_wtime();
	double result_cpp = Integral_cpp(0, 1, f);
	double t3 = omp_get_wtime();
	std::cout << "Integral_cpp(0,1) = " << result_cpp << std::endl;
	std::cout << (t3- t2)*1000 << " ms" << std::endl;
	std::cout << std::endl;

	double t12 = omp_get_wtime();
	double result = Integral_padding(0, 1, f);
	double t13 = omp_get_wtime();
	std::cout << "Integral_padding(0,1,f) = " << result << std::endl;
	std::cout << (t13 - t12) * 1000 << " ms" << std::endl;
	std::cout << std::endl;

	double t14 = omp_get_wtime();
	double result8 = Integral_Reduction(0, 1, f);
	double t15 = omp_get_wtime();
	std::cout << "Integral_Reduction(0,1,f) = " << result8 << std::endl;
	std::cout << (t15 - t14) * 1000 << " ms" << std::endl;
	std::cout << std::endl;*/
	std::cout << "==Padding=\n";
	auto v = run_experiment(Integral_padding);
	for (auto& r : v)
		std::cout << r << "\tSpeedup\t" << (v[0].time_ms / r.time_ms) << "\n";
	std::cout << "==REDUCTION==\n";
	 v = run_experiment(Integral_Reduction);
	for (auto& r : v)
		std::cout << r << "\tSpeedup\t" << (v[0].time_ms / r.time_ms) << "\n";
	std::cout << "==OMP==\n";
	v = run_experiment(Integral_omp);
	for (auto& r : v)
		std::cout << r << "\tSpeedup\t" << (v[0].time_ms / r.time_ms) << "\n";
	std::cout << "==MUTEX==\n";
	v = run_experiment(Integral_cpp);
	for (auto& r : v)
		std::cout << r << "\tSpeedup\t" << (v[0].time_ms / r.time_ms) << "\n";
	std::cout << "==FALSE==\n";
	v = run_experiment(Integral_razd);
	for (auto& r : v)
		std::cout << r << "\tSpeedup\t" << (v[0].time_ms / r.time_ms) << "\n";
	system("pause");
	return 0;
}


class C
{
	int m_val = 3;
public:
	C(int init) : m_val(init) {}
	int operator()(int x)
	{
		return m_val * x;
	}
};
