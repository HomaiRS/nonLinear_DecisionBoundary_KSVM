
In this computer project, we use kernel SVM to classify a dataset with two different labels. The data points are drawn randomly from uniform distribution with mean zero and variance 1. Using the given 𝑑" relationship in the assignment, data points will be labeled into two classes of C1 and C2 that have a highly non-linear decision boundary. Since the boundaries are not linear, we are not able to use many of the simple algorithm that we know so far such as perceptron training algorithm or other linear classifiers.

Therefore, we first lift data points using a kernel function regardless of the choice of feature maps to a higher dimensional space in the hope that data gets linearly separable. In this assignment, we use different kernel functions for mapping the data in the higher dimension and then we use support vector machine (SVM) algorithm to classify the data. In addition, we calculate the half spaces (decision boundaries) both analytically using the contours of discriminate functions as well as calculating decision boundaries numerically by approximation these functions.

I used radial basis kernel defined as <img src="https://render.githubusercontent.com/render/math?math=e^\frac{||X_i - x_j||}{1}">. Then, after setting the kernel, I solved the dual SVM problem which is a quadratic programing problem using “cvxopt” optimization solver implemented in Python. The summary of steps to solve dual SVM and reformulating its objective function matched with “cvxopt” solver format is defined as follows. For the detail of mathematical notations see <https://xavierbourretsicotte.github.io/SVM_implementation.html>.



1. Created ```P``` matrix where h<sub>&theta;</sub>(x) = &theta;<sub>o</sub> x + &theta;<sub>1</sub>x
<img src="https://render.githubusercontent.com/render/math?math=𝐻_ij = 𝑑_i 𝑑_j 𝐾(𝑥_i, 𝑥_j)">: coded as an outer product of ```d``` and kernel
2. Match the dualformulationwiththesolverformulation.(indicatedinequations1,2,3)
