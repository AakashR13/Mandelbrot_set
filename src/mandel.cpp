#include <iostream>
#include <complex>
#include <vector>
#include <chrono>
#include <functional>
#include <fstream>
#include <iostream>

#include "window.h"
#include "save_image.h"
#include "utils.h"
#include "params.h"

// clang++ -std=c++11 -stdlib=libc++ -O3 save_image.cpp utils.cpp mandel.cpp -lfreeimage

// Use an alias to simplify the use of complex type
using Complex = std::complex<double>;

// Convert a pixel coordinate to the complex domain
Complex scale(window<int> &scr, window<double> &fr, Complex c) {
	Complex aux(c.real() / (double)scr.width() * fr.width() + fr.x_min(),
		c.imag() / (double)scr.height() * fr.height() + fr.y_min());
	return aux;
}

// Check if a point is in the set or escapes to infinity, return the number if iterations
int escape(Complex c, const std::function<Complex( Complex, Complex)> &func) {
	Complex z(0);
	int iter = 0;

	while (abs(z) < 2.0 && iter < iter_max) {
		z = func(z, c);
		iter++;
	}

	return iter;
}

// Loop over each pixel from our image and check if the points associated with this pixel escape to infinity
void get_number_iterations(window<int> &scr, window<double> &fract, std::vector<int> &colors,
	const std::function<Complex( Complex, Complex)> &func) {
	int k = 0, progress = -1;
	for(int i = scr.y_min(); i < scr.y_max(); ++i) {
		for(int j = scr.x_min(); j < scr.x_max(); ++j) {
			Complex c((double)j, (double)i);
			c = scale(scr, fract, c);
			colors[k] = escape(c, func);
			k++;
		}
		// if(progress < (int)(i*100.0/scr.y_max())){
		// 	progress = (int)(i*100.0/scr.y_max());
		// 	std::cout << progress << "%\n";
		// }
	}
}

void fractal(window<int> &scr, window<double> &fract, std::vector<int> &colors,
	const std::function<Complex( Complex, Complex)> &func, const char *fname, bool smooth_color) {
	auto start = std::chrono::steady_clock::now();
	get_number_iterations(scr, fract, colors, func);
	auto end = std::chrono::steady_clock::now();
	std::cout << "Time to generate " << fname << " = " << std::chrono::duration <double, std::milli> (end - start).count() << " [ms]" << std::endl;
	
	// Save (show) the result as an image
	plot(scr, colors, iter_max, fname, smooth_color);

	// Output details to a text file
	std::ofstream out("./res/fractal_details_CPU.txt", std::ios::app);
	if (out.is_open()) {
		out << "File name: " << fname << "\n";
		out << "GPU Accelerated: false" << "\n";
		out << "Time to generate: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms\n";
		out << "Iterations: " << iter_max << "\n";
		out << "Smooth color: " << (smooth_color ? "true" : "false") << "\n";
		out << "----------------------------------------\n";
		out.close();
	} else {
		std::cerr << "Unable to open file for writing" << std::endl;
	}
}

void mandelbrot() {
	// Define the size of the image
	window<int> scr(0, win_width, 0, win_height);
	// The domain in which we test for points
	window<double> fract(-2.2, 1.2, -1.7, 1.7);

	// The function used to calculate the fractal
	auto func = [] (Complex z, Complex c) -> Complex {return z * z + c; };

	const char *fname = "./res/mandelbrot_nonaccel.png";
	bool smooth_color = true;
	std::vector<int> colors(scr.size());

	// Experimental zoom (bugs ?). This will modify the fract window (the domain in which we calculate the fractal function) 
	//zoom(1.0, -1.225, -1.22, 0.15, 0.16, fract); //Z2
	
	fractal(scr, fract, colors, func, fname, smooth_color);
}

void triple_mandelbrot() {
	// Define the size of the image
	window<int> scr(0, win_width, 0, win_height);
	// The domain in which we test for points
	window<double> fract(-1.5, 1.5, -1.5, 1.5);

	// The function used to calculate the fractal
	auto func = [] (Complex z, Complex c) -> Complex {return z * z * z + c; };

	const char *fname = "./res/triple_mandelbrot_nonaccel.png";
	bool smooth_color = true;
	std::vector<int> colors(scr.size());

	fractal(scr, fract, colors, func, fname, smooth_color);
}

int main() {

	mandelbrot();
	triple_mandelbrot();

	return 0;
}
