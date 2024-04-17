#include <iostream>
#include <mlpack.hpp>
#include <string>
#include <mlpack/core.hpp>
#include <mlpack/methods/gmm.hpp>
#include <mlpack/methods/dbscan/dbscan_impl.hpp>
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/core/dists/gaussian_distribution.hpp>

using namespace mlpack;

int main() {

	// Appareantly DataLoader<> exists where you may load a model and set training and prediction easily but you need to build it.

	

	/* 1. Load the appropriate dataset
	MAKE SURE TO EXTRACT FEATURES OR TEXT BASED CHARS!!!
	OTHERWISE IT WILL MESS UP YOUR DATA.
	RESULT: 0-VALUE DATA AND TRANSPOSE IS SOMEHOW OMITTED AS WELL.
	Tip: set the third argument to true to see if your dataset loaded successfully*/

std::cout << "NR. 1" << std::endl;

	arma::Mat<double> dataset;
	data::Load("iris.csv", dataset, true, true);
	/* Fitting
	auto scaler = mlpack::data::ScalingModel(STANDARD_SCALER);
	scaler.Fit(dataset&);
	https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/preprocess/scaling_model.hpp
	*/
	arma::mat scaled_dataset;
	mlpack::data::StandardScaler scaler;
	scaler.Fit(dataset);
	scaler.Transform(dataset,scaled_dataset);

	/*Check this function to normalize labels
	https://github.com/mlpack/mlpack/blob/master/src/mlpack/core/data/normalize_labels.hpp*/
	arma::Row<size_t> labels;
	/* Casting last column which is a matrix to a row-vector and delete it from the matrix
	 labels = conv_to<arma::Row<size_t>>::from(dataset.row(dataset.n_rows-1));
	 dataset.shed_row(dataset.n_rows-1);
	 Would've been relevant if the normalized labels was inside the dataset
	 https://github.com/mlpack/mlpack/blob/master/src/mlpack/core/util/conv_to.hpp*/
	data::Load("iris_labels.txt", labels);

	/* Declare for splitting 
	https://github.com/mlpack/mlpack/blob/master/src/mlpack/core/data/split_data.hpp*/
	arma::Row<size_t> train_labels, test_labels;
	arma::mat train_data, test_data;
	mlpack::data::StratifiedSplit(dataset, labels, train_data, test_data, train_labels, test_labels, .2, true);



	/* 
	2. Understand measure the mean, median and mode
	 measure the variance, standard deviation and interquartile rande.
	 
	 Assessing the assymetry and shape of the distribution

	 Visualize the distribution och each feature using histograms or density plots.
	 That gives me the frequency or probability density of different values or ranges of values within each feature

	 Q-Q plot. comparing the dist of each feature to a known theoretical distribution eg normal dist using Q-Q plot

	 Identify extreme or unusual values that may deviate significantly from the rest of the data.
	*/
	/* 
	takes a vector, matrix or qube. matrix and qube may 
	have dim as second parameter although dim=0 is assumed which is column. 
	because we want rows as data points and the matrix is transposed when loaded. 
	*/
std::cout << "NR. 2" << std::endl;

	arma::mat mean = arma::mean(scaled_dataset);

	/* 
	median takes vector and matrix. matrix takes dim as well 
	*/

	arma::mat median = arma::median(scaled_dataset);

	/* 
	deviation takes vector and matrix with 1-3 args and matrix 1-2 args
	vector = stddev(v, norm_type)
	matrix = stddev(m, norm_type, dim)
	norm_type=0(DEFAULT) refers to normalising using N-1 where N is the number of samples which provides the best unbiased estimator
	norm_type=1 performs normalising using N, which provides the second moment around the mean 
	*/
	arma::mat standard_deviation = arma::stddev(scaled_dataset);

	/* variance takes a vector and a matrix with the args
	vector = var(v, norm_type) refer to deviation for norm_type description
	matrix = var(m, norm_type, dim) refer to deviation for norm_type and mean for dim description
	*/
	arma::mat variance = arma::var(scaled_dataset);

	/* 3. Determine the amount of comp/clusters. 
	Try BIC(fun fact. Developed by Gideon E. Schwarz and published in 1978 paper), 
	 or AIC(fun fact. Formulated by Hirotsugu Akaike first time 1971) 
	 or cross-validation if unknown. 
	Iterate over different amount of components 
	and evaluate each models performance using the chosen criterion.

	Appareantly mlpack does have a built-in function for this named:
	DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

	If there is RAM issues i.e. if the dataset is very large or epsilon is large,
	you may set the batchMode to false.
	When so, each point will be searched iteratively which could be slower
	but will use less memory.
	https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/dbscan/dbscan.hpp
	*/

std::cout << "NR. 3" << std::endl;

// Please not that this section is not fully done. This should be the closest to 
// BIC/AIC but the correct value needs to be generated or extracted.

	arma::mat centroids;
	arma::Row<size_t> assignments;

	mlpack::dbscan::DBSCAN<> db_obj(.00005,1);	
	
	UnionFind nr_of_gaussians = mlpack::UnionFind(0);
	// Generating assignments and centroids
	db_obj.Cluster(scaled_dataset, assignments, centroids);
	

	/* 4. Initialize the GMM with the components.
	Train the GMM using the preprocessed dataset.
	This involves estimating the parameters means, covariance and mixture weights of the Gaussian components.
	https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/gmm/gmm.hpp
	
	Weight initilization? 
	https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/perceptron/initialization_methods/random_init.hpp
	*/
std::cout << "NR. 4" << std::endl;

size_t gaussians = 3;
size_t dimensions = 4;


/*
mlpack::EMFit<> fitting(
	const size_t maxIterations = 300,
    const double tolerance = 1e-10, 
mlpack::KMeans<>(
	size_t maxIterations = 1000UL, 
	mlpack::EuclideanDistance metric = mlpack::EuclideanDistance(), 
	mlpack::SampleInitialization partitioner = mlpack::SampleInitialization(), 
	mlpack::MaxVarianceNewCluster emptyClusterAction = mlpack::MaxVarianceNewCluster()
	), 
mlpack::PositiveDefiniteConstraint()
);
*/


GMM gmm(gaussians, dimensions);
gmm.Train(train_data, 1000, false /*, fitting */);



	/* 5. Evaluate the trained GMM models performance using appropriate metrics.
	Compute the likelihood of the data points under the GMM model.
	Optionally validate the models performance using a seperate validation dataset or cross-validation 
	https://github.com/mlpack/mlpack/blob/master/src/mlpack/core/cv/metrics/accuracy.hpp
	*/
std::cout << "NR. 5" << std::endl;

	double accuracy = mlpack::Accuracy::Evaluate(gmm, test_data, test_labels);

	std::cout << "Accuracy: " << accuracy << "%" << std::endl;

	
	/* Save your model with the formats: json, xml or binary with function signature:
	mlpack::data::Save("filename.format", "name of the model", model-object, "format")
 
	Appareantly you're obligated to have the filetype included in the filename
	due to mlpack being unable to detect the filetype even tho we specify it?.*/
	// mlpack::data::Save("iris_gmm.json", "iris_gmm", gmm, "json");

	// std::cout << "Size of test data and labels: " << arma::size(test_data) << " " << arma::size(test_labels) << std::endl;

	return 0;
}
