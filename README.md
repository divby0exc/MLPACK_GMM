# GMM Example
## GMM Training and Evaluation

[![N|Solid](https://images.g2crowd.com/uploads/product/image/social_landscape/social_landscape_462d2acd8cbaecfd8a02dab9241bc907/mlpack.png)](https://www.mlpack.org/)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

==TODO: Add a function to parse a dataset to save and delete labels and features from==

GMM code example using mlpack on how the data needs to be formated to be worked with,
trained and evaluating it's accuracy.

I hope this will help you get started with mlpack and the GMM model as fast as possible.
If you notice anything wrong or something that should be added by any reason.
Please contact me and thanks for reading.

 >My example is built in Ubuntu 22.04.4

## Features

- How to install mlpack and it's dependencies
- How to initialize the GMM
- How to read in your dataset
- How to train and evaluate your model
- How to customize your GMM
- How to build your code
- Preprocessing


## Dependencies

- [C++ Compiler] - g++
- [mlpack](https://github.com/mlpack/mlpack) - Header-only lib
- [OpenMP] - Optimizing matrice work
- [Armadillo] - Provides necessary datatypes
- [OpenBLAS] - Used by the train function
- [LAPACK] - Matrix decomposition
- [CMake] - Bulding the source
- [Matplotlib-cpp] - Optional
- [Python3] - Used for matplotlib-cpp
- [Numpy]

## Installation

- mlpack      >=Cxx14 compiler
- Armadillo   >= 9.800
- ensmallen   >= 2.10.0
- cereal      >= 1.1.2
- matplotlib-cpp default: python3

#### 1. Install the dependencies before you build mlpack

```sh
sudo apt-get install libarmadillo-dev libensmallen-dev libcereal-dev libstb-dev g++ cmake
```

#### 2. Clone the mlpack repo

```sh
git clone https://github.com/mlpack/mlpack.git
```

#### 3. Build mlpack from the root source

```sh
mkdir build && cd build/
cmake ..
sudo make install
```

CMake >= v3.14.0
```sh
cmake -S . -B build
sudo cmake --build build --target install
```

==Once done you only need to include the header in your src to get started:==
```cpp
#include <mlpack.hpp>
```

Although you will need more headers depending on your goal.

#### Optional 4. Clone and build matplotlib-cpp
matplotlib cmake is by default configured for python3 but is compatible with python2
==For other ways to install matplotlib-cpp, please press the "matplotlib-cpp" link below==
```sh
sudo apt-get install python3-matplotlib python3-numpy python3.10-dev
```
Please refer to [matplotlib-cpp](https://github.com/lava/matplotlib-cpp) repo for more information regarding usage
Although I will leave an example snippet here:
```cpp
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
int main() {
    plt::plot({1,3,2,4});
    plt::show();
}
```

## Build args:
```sh
g++ -fopenmp -o output_name filename.cpp -O2 -larmadillo -llapack -lblas
```

If you want to ==plot== in C++ you may add this arg:
==Note:== make sure your python and matplotlib-cpp path is correct
```sh
-I/usr/include/python3.10 -lpython3.10 -I/usr/local/matplotlib-cpp
```

## How matrices works in mlpack
> Armadillo is column-major compared to numpy which is row-major.
> Which means that the ==columns is contiguous in memory not the rows==.
> Your dataset will get ==tranposed== when you load it leading to
> Observations/Data points as Columns and Dimensions as rows.

You will load your dataset using mlpacks load function where:
filename is a string, matrix is usually your arma::mat dataset and the rest may be omitted if desired. 
>Although I will recommend you to set fatal=true due to probable failure in the loading which you won't find out until later if false.

==!IMPORTANT!==
You need to extract your labels and features before loading your dataset.
At the moment. I haven't found a way to encode it after loaded.
If there is a NaN value in e.g. a data-point/column it will omit the whole column.

If you load it with text like this
```cpp
if(!(data::Load("iris.csv",dataset)
    std::cout << "Dataset didn't load correctly\n";
```
```sh
Dataset didn't load correct
-====================WITH TEXT===================
   5.1000   3.5000   1.4000   0.2000        0
        0        0        0        0        0
        0        0        0        0        0
        0        0        0        0        0
        0        0        0        0        0
        0        0        0        0        0
        0        0        0        0        0
```
I found a class named DatasetMapper which I thought solved my issue
```cpp
data::DatasetInfo info;
	if(!(data::Load("orig_iris.csv", with_text, info)))
		std::cout << "Dataset didn't load correct" << std::endl;
```
It encodes the non-numeric like you would want to with the labels. But perhaps not all the data right?
```sh
====================WITH TEXT===================
         0    1.0000    2.0000    3.0000    4.0000    5.0000    6.0000    4.0000    5.0000    7.0000    2.0000    6.0000    8.0000    8.0000    9.0000   10.0000   11.0000    6.0000    1.0000   11.0000    1.0000    6.0000    1.0000    4.0000    1.0000    8.0000    5.0000    5.0000   12.0000   12.0000    3.0000    8.0000    6.0000   12.0000   13.0000    2.0000    5.0000   13.0000    2.0000    7.0000    1.0000    5.0000   14.0000    7.0000    5.0000    1.0000    8.0000    1.0000    4.0000   15.0000    5.0000   16.0000   17.0000   18.0000   13.0000   19.0000   11.0000   20.0000    2.0000   21.0000   12.0000
```
So at last. If you extract the features and labels into different files and only load your dataset containing numerics. The results are this:
```sh
==========================WITHOUT TEXT======================
   5.1000   4.9000   4.7000   4.6000   5.0000   5.4000   4.6000   5.0000   4.4000   4.9000   5.4000   4.8000   4.8000   4.3000   5.8000   5.7000   5.4000   5.1000   5.7000   5.1000   5.4000   5.1000   4.6000   5.1000   4.8000   5.0000   5.0000   5.2000   5.2000   4.7000   4.8000   5.4000   5.2000   5.5000   4.9000   5.0000   5.5000   4.9000   4.4000   5.1000   5.0000   4.5000   4.4000   5.0000   5.1000   4.8000   5.1000   4.6000   5.3000   5.0000   7.0000   6.4000   6.9000   5.5000   6.5000   5.7000   6.3000   4.9000   6.6000   5.2000   5.0000   5.9000   6.0000   6.1000   5.6000   6.7000   5.6000   5.8000
```

Mlpack way:
```cpp
data::Load(filename, matrix, fatal=false, transpose=true, type=FileType::AutoDetect)
```

Where matrix/dataset is one of following:
```cpp
mat	 = 	Mat<double>
dmat	 = 	Mat<double>
fmat	 = 	Mat<float>
cx_mat	 = 	Mat<cx_double>
cx_dmat	 = 	Mat<cx_double>
cx_fmat	 = 	Mat<cx_float>
umat	 = 	Mat<uword>
imat	 = 	Mat<sword>

e.g.
arma::mat my_matrix;
mlpack::data::Load("dataset.csv", my_matrix);
```

You also have col-vectors 1-column:
vec and colvec are the same. Only different aliases for convenience.
```cpp
vec	 = 	colvec	 = 	Col<double>
dvec	 = 	dcolvec	 = 	Col<double>
fvec	 = 	fcolvec	 = 	Col<float>
cx_vec	 = 	cx_colvec	 = 	Col<cx_double>
cx_dvec	 = 	cx_dcolvec	 = 	Col<cx_double>
cx_fvec	 = 	cx_fcolvec	 = 	Col<cx_float>
uvec	 = 	ucolvec	 = 	Col<uword>
ivec	 = 	icolvec	 = 	Col<sword>
```

Row-vector 1-row:
```cpp
rowvec	 = 	Row<double>
drowvec	 = 	Row<double>
frowvec	 = 	Row<float>
cx_rowvec	 = 	Row<cx_double>
cx_drowvec	 = 	Row<cx_double>
cx_frowvec	 = 	Row<cx_float>
urowvec	 = 	Row<uword>
irowvec	 = 	Row<sword>
```

Armadillo way:
```cpp
arma::mat dataset;
dataset.load("dataset.filetype");
```

## Preprocessing
#### [Normalizing](https://github.com/mlpack/mlpack/tree/master/src/mlpack/core/data/scaler_methods)
We have 6 different types with the same function signature
Make sure to preceed your method with the path below:

> <mlpack/core/data/scaler_methods/ <normalization method> >

```cpp
#include "max_abs_scaler.hpp" > MaxAbsScaler
#include "mean_normalization.hpp" > MeanNormalization
#include "min_max_scaler.hpp" > MinMaxScaler
#include "pca_whitening.hpp" > PCAWhitening
#include "standard_scaler.hpp" > StandardScaler
#include "zca_whitening.hpp" > ZCAWhitening
```

Example:
```cpp
arma::mat input;
Load("train.csv", input);
arma::mat output;

StandardScaler scaler;
scaler.Fit(input)

scaler.Transform(input, output);

Retransform the input.
scaler.InverseTransform(output, input);
```

## Train the GMM
Simplest way to get started is initializing your GMM as follow:
```cpp
#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/gmm.hpp>

GMM gmm(gaussians, dimensions);
gmm.Train(train_data, 1000);

double accuracy = mlpack::Accuracy::Evaluate(gmm, test_data, test_labels);
```

In the above-mentioned accuracy function. Underlying function is using the gmm.Classify() function instead of calling the gmm.Classify() and calculates the accuracy for you.

The GMM, once trained, can be used to generate random points from the
 * distribution and estimate the probability of points being from the distribution.  
 * The parameters of the GMM can be obtained through the
 * accessors and mutators.
 
E.g. functions:
```cpp
 // Get the probability of 'observation' being observed from this GMM.
 double probability = gmm.Probability(observation);
 
 // Get a random observation from the GMM.
 arma::vec observation = gmm.Random();
 ```
 
 E.g. of extracting properties
 ```cpp
 // 2 is the component/gaussian you want to access.
 arma::vec mean = gmm.Means()[2];
 arma::mat covariance = gmm.Covariances()[2];
 double prior_weight = gmm.Weights()[2];
```

#### Fitting
The Train() method uses a template type 'FittingType'.  The FittingType template class must provide a way for the GMM to train on data.

* @tparam FittingType The type of fitting method which should be used 
*       (EMFit<> is suggested).
* @param observations Observations of the model.
* @param trials Number of trials to perform; the model in these trials with
*      the greatest log-likelihood will be selected.
* @param useExistingModel If true, the existing model is used as an initial
*      model for the estimation.
* @param fitter The fitter to use, optional.
* @return The log-likelihood of the best fit.
---> [EMFit](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/gmm/em_fit.hpp) <---
 
```cpp
  template<typename FittingType = EMFit<>>
  double Train(const arma::mat& observations,
               const size_t trials = 1,
               const bool useExistingModel = false,
               FittingType fitter = FittingType());
```

```cpp
// This is how it looks
gmm.Train<EMFit<KMeans<>>>(train_data, 1000);
// Although the template <> part is assumed/default fittingtype and may be omitted.
// This is the same as above-mentioned.
gmm.Train(train_data, 1000);
```

Full definition of the EMFit method if you want to modify any parameters or store state:
```cpp
mlpack::EMFit<> fitting(
        const size_t maxIterations = 300,
        const double tolerance = 1e-10,
        InitialClusteringType clusterer = InitialClusteringType(),
        CovarianceConstraintPolicy constraint = CovarianceConstraintPolicy());
```
=======================================================================

==InitialClusteringType() comes from:==

[KMeans](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/kmeans/kmeans.hpp)

Four template parameters can (optionally) be supplied: the distance metric to
use, the policy for how to find the initial partition of the data, the
actions to be taken when an empty cluster is encountered, and the
implementation of a single Lloyd step to use.
 
```cpp
mlpack::KMeans<>(
	size_t maxIterations = 1000UL, 
	mlpack::EuclideanDistance metric = mlpack::EuclideanDistance(), 
	mlpack::SampleInitialization partitioner = mlpack::SampleInitialization(), 
	mlpack::MaxVarianceNewCluster emptyClusterAction = mlpack::MaxVarianceNewCluster()
	template<class, class> class LloydStepType = NaiveKMeans,
         typename MatType = arma::mat>
	);
```

Where distance metric is one of these and comes from the mlpack namespace:
These 4 are part of [LMetric](https://github.com/mlpack/mlpack/blob/master/src/mlpack/core/metrics/lmetric.hpp)
- ManhattanDistance
- EuclideanDistance
- SquaredEuclideanDistance
- ChebyshevDistance
- [MahalanobisDistance<>](https://github.com/mlpack/mlpack/blob/master/src/mlpack/core/metrics/mahalanobis_distance.hpp)
- [IPMetric](https://github.com/mlpack/mlpack/blob/master/src/mlpack/core/metrics/ip_metric.hpp) (requires a [KernelType](https://github.com/mlpack/mlpack/blob/master/doc/developer/kernels.md) parameter)

Partitioner, variance and lloyd-steps are included through this:
```cpp
#include <mlpack/methods/kmeans/kmeans.hpp>
```

Where initialization partitioner is one of these and comes from the mlpack namespace:
- [SampleInitialization](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/kmeans/sample_initialization.hpp)
- [KMeansPlusPlusInitialization](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/kmeans/kmeans_plus_plus_initialization.hpp)
- [RandomPartition](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/kmeans/random_partition.hpp)

Where variance cluster is one of these and comes from the mlpack namespace:
- [MaxVarianceNewCluster](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/kmeans/max_variance_new_cluster.hpp)
- [KillEmptyClusters](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/kmeans/kill_empty_clusters.hpp)
- [AllowEmptyClusters](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/kmeans/allow_empty_clusters.hpp)


==There is a Lloyd-step to modify as well==
-	NaiveKMeans
-	DualTreeKMeans
-	ElkanKMeans
-	HamerlyKMeans
-	PellegMooreKMeans

Full implementation of the clustering type is as followed:
You can do it in two ways.
First is through the template declaration. Beware you may not set the iteration in the template. You will set it as argument.

```cpp
kmeans::KMeans<
	mlpack::EuclideanDistance,
	mlpack::SampleInitialization,
	mlpack::MaxVarianceNewCluster,
	mlpack::PellegMooreKMeans> k(1000);
```

If you want to do it through parameters, NaiveKMeans is assumed and looks like this:

```cpp
kmeans::KMeans<> k(1000, EuclideanDistance(), SampleInitialization(), MaxVarianceNewCluster());
```

=======================================================================

==CovarianceConstraintPolicy() comes from:==

> mlpack namespace

All the constraints are included through
```cpp
#include <mlpack/methods/gmm/gmm.hpp>
```


 Given a covariance matrix, force the matrix to be positive definite.  Also
 force a minimum value on the diagonal, so that even if the matrix is
 invertible, it doesn't cause problems with Cholesky decompositions.  The
 forcing here is also done in order to bring the condition number of the
 matrix under 1e5 (10k), which should help with numerical stability.

- [PositiveDefiniteConstraint](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/gmm/positive_definite_constraint.hpp)

This class enforces no constraint on the covariance matrix.  It's faster this
way, although depending on your situation you may end up with a
 non-invertible covariance matrix.
 
- [NoConstraint](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/gmm/no_constraint.hpp)

Force a covariance matrix to be diagonal.

- [DiagonalConstraint](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/gmm/diagonal_constraint.hpp)

Given a vector of eigenvalue ratios, ensure that the covariance matrix always
has those eigenvalue ratios.  When you create this object, make sure that the
vector of ratios that you pass does not go out of scope, because this object
holds a reference to that vector instead of copying it.  (This doesn't apply
if you are deserializing the object from a file.)

[EigenvalueRatioConstraint](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/gmm/eigenvalue_ratio_constraint.hpp)

=======================================================================

#### Classification

Classify the given observations as being from an individual component in
this GMM.  The resultant classifications are stored in the 'labels' object,
and each label will be between 0 and (Gaussians() - 1).  Supposing that a
point was classified with label 2, and that our GMM object was called
'gmm', one could access the relevant Gaussian distribution as follows:

@param observations List of observations to classify.
@param labels Object which will be filled with labels.

```cpp
void Classify(const arma::mat& observations,
                arma::Row<size_t>& labels) const;
```

Example:
```cpp
GMM gmm(3,4);
arma::Row<size_t> labels;
arma::mat our_dataset;

gmm.Classify(our_dataset, labels);
```

## License

MIT

@article{mlpack2023,
    title     = {mlpack 4: a fast, header-only C++ machine learning library},
    author    = {Ryan R. Curtin and Marcus Edel and Omar Shrit and 
                 Shubham Agrawal and Suryoday Basak and James J. Balamuta and 
                 Ryan Birmingham and Kartik Dutt and Dirk Eddelbuettel and 
                 Rishabh Garg and Shikhar Jaiswal and Aakash Kaushik and 
                 Sangyeon Kim and Anjishnu Mukherjee and Nanubala Gnana Sai and 
                 Nippun Sharma and Yashwant Singh Parihar and Roshan Swain and 
                 Conrad Sanderson},
    journal   = {Journal of Open Source Software},
    volume    = {8},
    number    = {82},
    pages     = {5026},
    year      = {2023},
    doi       = {10.21105/joss.05026},
    url       = {https://doi.org/10.21105/joss.05026}
}
