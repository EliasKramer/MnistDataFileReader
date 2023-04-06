#include "training_data.hpp"
#include <iostream>
#include <chrono>
int main()
{
	std::cout << "Hello World!" << std::endl;

	const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	std::cout << std::endl << "Reading files..." << std::endl;
	
	digit_image_collection_t training_data_mnist = load_mnist_data(
		"/data/train-images.idx3-ubyte",
		"/data/train-labels.idx1-ubyte");
	
	digit_image_collection_t testing_data_mnist = load_mnist_data(
		"/data/t10k-images.idx3-ubyte",
		"/data/t10k-labels.idx1-ubyte");

	const std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
	std::cout
		<< "Reading files done. took : "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms" << std::endl;

	return 0;
}