
In this computer project, we use kernel SVM to classify a dataset with two different labels. The data points are drawn randomly from uniform distribution with mean zero and variance 1. Using the given <img src="https://render.githubusercontent.com/render/math?math=d_i"> relationship in the assignment, data points will be labeled into two classes of C1 and C2 that have a highly non-linear decision boundary. Since the boundaries are not linear, we are not able to use perceptron training algorithm or other linear classifiers.

Therefore, we first lift data points using a kernel function regardless of the choice of feature maps to a higher dimensional space in the hope that data gets linearly separable. In this repo, we use a kernel function for mapping the data into the higher dimension and then we use support vector machine (SVM) algorithm to classify the data in that space. In addition, we calculate the half spaces (decision boundaries) both analytically using the projection of contours of discriminate functions and numerically using approximation of these functions.

I used radial basis kernel defined as <img src="https://render.githubusercontent.com/render/math?math=e^\frac{||X_i - x_j||}{1}">. Next, I solved the dual SVM problem which is a quadratic programing problem using “cvxopt” optimization solver implemented in Python. The summary of steps to solve dual SVM and reformulating its objective function matched with “cvxopt” solver format is defined as follows. For the detail of mathematical notations see <https://xavierbourretsicotte.github.io/SVM_implementation.html>.

<img width="702" alt="Formulation2" src="https://user-images.githubusercontent.com/43753085/104140975-f74b7d80-5379-11eb-81ba-669e413a4c01.png">

The dual SVM in most cases (e.g., 300 data points) is solvable efficiently and fast in a few seconds. In almost all cases that I ran, problem converges to the optima solutions <img src="https://render.githubusercontent.com/render/math?math=\alpha_i"> in less than 15 iterations. I also tried the polynomial kernel, but the radial basis kernel classifies the data more accurately as it can detect the decision boundaries more accurately. Therefore, I only included the results where I used RBF kernels. In the following figures, green crosses are input patterns with desired class 1, and the purple diamonds are the input patterns with desired class −1. The decision boundary H (the blue line) can perfectly separate the two classes that were otherwise not linearly separable. Support vectors ared marked by circles. As expected, there are no patterns in between the “guard lines” H+ and H−.

<img width="959" alt="allFig1" src="https://user-images.githubusercontent.com/43753085/104141034-3679ce80-537a-11eb-965f-de6ce1510e09.png">

---

<img width="963" alt="allFig2" src="https://user-images.githubusercontent.com/43753085/104141088-86589580-537a-11eb-8fd4-918997952128.png">

---

<img width="985" alt="allFig3" src="https://user-images.githubusercontent.com/43753085/104141187-1b5b8e80-537b-11eb-8022-f45cc6acaee0.png">
