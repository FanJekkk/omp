#include <vector>
#include <thread>
#include <omp.h>
#include <mutex>
#include <stdio.h>

#define STEPS 100000

double f(double x)
{
	return 4 / (1 + x*x);
}
double Integral_omp(double a, double b)
{
	auto T = std::thread::hardware_concurrency();
	double dx = (b - a) / STEPS;
	std::vector<std::thread>threads;
	double res = 0;
	for (unsigned t = 0; t < T; ++t)
		threads.emplace_back(std::thread([T, t, dx, &res, a]()
	{
		for (auto i = t; i < STEPS; i += T)
			res += f((i + 0.5)*dx + a);
	}));
	for (auto&thr : threads)
		thr.join();
	return res*dx;
#pragma omp parallel
	{
		int t = omp_get_thread_num();
		double r = 0;//частичн. суммы
for (int i = t; i<STEPS; i += T)
	r += f(dx*(i + 0.5) + a);
res += r;
#pragma omp critical
{
	res += r;
}
}
return res*dx;
}

double Integral_cpp(double a, double b)
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
				r += f(dx*((double)i + 0.5) + a);
			mtx.lock();
			res += r;
			mtx.unlock();
		}));
	}
	for (auto &thr : threads)
		thr.join();
	return res*dx;
}

int main(int argc, char** argv)
{
	double t2 = omp_get_wtime();
	double result_cpp = Integral_cpp(0, 1);
	double t3 = omp_get_wtime();
	printf("result: %g,time: %g\n", result_cpp, (t3 - t2) * 1000);
	return 0;
}


class C
{
	int m_val = 3;
public:
	C(int init) : m_val(init) {}
	int operator()(int x)
	{
		return m_val*x;
	}
};


