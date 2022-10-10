# multi-drone-quiz

The Euclidean Signed Distance Transform (ESDF) has been implemented for finding out the closest obstacles from each point.

The algorithm given in http://people.cs.uchicago.edu/~pff/papers/dt.pdf has been parallelized (i.e., the algorithm runs on multiple rows/columns simultaneously) for efficient implementation.

Numpy arrays of higher dimensions have been used instead of integers/ lists. The code has been commented accordingly to show the corresponding modifications. For example, in the variable `k`, arrays have been used instead of integers to store the rightmost parabola points of all the columns. The elements are accessed and modified based on the value of `k` for each of the columns.