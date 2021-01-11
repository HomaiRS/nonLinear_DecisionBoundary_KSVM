
In this computer project, we use kernel SVM to classify a dataset with two different labels. The data points are drawn randomly from uniform distribution with mean zero and variance 1. Using the given 𝑑" relationship in the assignment, data points will be labeled into two classes of C1 and C2 that have a highly non-linear decision boundary. Since the boundaries are not linear, we are not able to use many of the simple algorithm that we know so far such as perceptron training algorithm or other linear classifiers.

Therefore, we first lift data points using a kernel function regardless of the choice of feature maps to a higher dimensional space in the hope that data gets linearly separable. In this assignment, we use different kernel functions for mapping the data in the higher dimension and then we use support vector machine (SVM) algorithm to classify the data. In addition, we calculate the half spaces (decision boundaries) both analytically using the contours of discriminate functions as well as calculating decision boundaries numerically by approximation these functions.

I used radial basis kernel defined as <img src="https://render.githubusercontent.com/render/math?math=e^\frac{||X_i - x_j||}{1}">. Then, after setting the kernel, I solved the dual SVM problem which is a quadratic programing problem using “cvxopt” optimization solver implemented in Python. The summary of steps to solve dual SVM and reformulating its objective function matched with “cvxopt” solver format is defined as follows. For the detail of mathematical notations see <https://xavierbourretsicotte.github.io/SVM_implementation.html>.

<img width="702" alt="Formulation2" src="https://user-images.githubusercontent.com/43753085/104140975-f74b7d80-5379-11eb-81ba-669e413a4c01.png">

The dual SVM in most cases using even 300 data points is solved in several seconds and it is pretty fast and efficient. In almost all cases that I ran, problem converges to the optima solutions (𝛼") in less than 15 iterations. I also used polynomial kernel (as indicated in the appendix in my code). However, I got the best decision boundaries using radial basis kernel. Thus, I only included the results in which I used for RBF kernel. In Figure 1, green crosses are input patterns with desired class 1, and the purple diamonds are the input patterns with desired class −1. The decision boundary H (the blue line) can perfectly separate the two classes that were otherwise not linearly separable. Support vectors marked by circles. As expected, there are no patterns in
between the “guard lines” H+ and H−.

<img width="959" alt="allFig1" src="https://user-images.githubusercontent.com/43753085/104141034-3679ce80-537a-11eb-965f-de6ce1510e09.png">

<img width="969" alt="allFig2" src="https://user-images.githubusercontent.com/43753085/104141063-5dd09b80-537a-11eb-884c-faf63dc905ec.png">
