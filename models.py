import nn

class PerceptronModel(object):
    def __init__(self, dim):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dim` is the dimensionality of the data.
        For example, dim=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dim)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x_point):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x_point: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x_point)

    def get_prediction(self, x_point):
        """
        Calculates the predicted class for a single data point `x_point`.

        Returns: -1 or 1
        """
        "*** YOUR CODE HERE ***"
        dotProduct = nn.as_scalar(self.run(x_point))
        if dotProduct >= 0:
            return 1
        else:
            return -1
        

    def train_model(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1

        perfectAcc = False
        while perfectAcc != True:
            perfectAcc = True
            for x, y in dataset.iterate_once(batch_size):
                predictedY = self.get_prediction(x)
                if predictedY != nn.as_scalar(y):
                    self.w.update(-1*predictedY, x)
                    perfectAcc = False        

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        hidden_layer_size = 300
        self.w1 = nn.Parameter(1, hidden_layer_size)
        self.b1 = nn.Parameter(1, hidden_layer_size)

        self.w2 = nn.Parameter(hidden_layer_size,1)
        self.b2 = nn.Parameter(1,1)

        self.learning_rate = 0.05

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        xm = nn.Linear(x, self.w1)
        layer1output = nn.AddBias(xm, self.b1)

        layer2input = nn.ReLU(layer1output)

        ym = nn.Linear(layer2input, self.w2)
        predicted_y = nn.AddBias(ym, self.b2)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y, y)

    def train_model(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 10

        perfectAcc = False
        currAvgLoss = 1
        while currAvgLoss >= 0.005:
            perfectAcc = True
            currLosses = []
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                currLosses.append(nn.as_scalar(loss))

                grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2 = nn.gradients([self.w1, self.b1, self.w2, self.b2], loss)
                self.w1.update(-1*self.learning_rate, grad_wrt_w1)
                self.b1.update(-1*self.learning_rate, grad_wrt_b1)
                self.w2.update(-1*self.learning_rate, grad_wrt_w2)
                self.b2.update(-1*self.learning_rate, grad_wrt_b2)

            lossSum = 0
            for i in range(0, len(currLosses)):
                lossSum += currLosses[i]
            currAvgLoss = lossSum/len(currLosses)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        hidden_layer_size1 = 200
        hidden_layer_size2 = 50
        self.w1 = nn.Parameter(784, hidden_layer_size1)
        self.b1 = nn.Parameter(1, hidden_layer_size1)
        self.w2 = nn.Parameter(hidden_layer_size1, hidden_layer_size2)
        self.b2 = nn.Parameter(1, hidden_layer_size2)
        self.w3 = nn.Parameter(hidden_layer_size2, 10)
        self.b3 = nn.Parameter(1, 10)
        self.learning_rate = 0.06
        self.epochs = 20
        self.batch_size = 10
        print(f'Learning rate: {self.learning_rate}, Batch size: {self.batch_size}, Hidden layer sizes: {hidden_layer_size1} and {hidden_layer_size2}, Number of epochs {self.epochs}\n')

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        xm1 = nn.Linear(x, self.w1)
        xm1_bias = nn.AddBias(xm1, self.b1)
        relu1 = nn.ReLU(xm1_bias)
        xm2 = nn.Linear(relu1, self.w2)
        xm2_bias = nn.AddBias(xm2, self.b2)
        relu2 = nn.ReLU(xm2_bias)
        ym = nn.Linear(relu2, self.w3)
        return nn.AddBias(ym, self.b3)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y, y)

    def train_model(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for epoch in range(self.epochs):
            net_loss = 0
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients([self.w1, self.b1, self.w2, self.b2, self.w3, self.b3], loss)

                self.w1.update(-self.learning_rate, gradients[0])
                self.b1.update(-self.learning_rate, gradients[1])
                self.w2.update(-self.learning_rate, gradients[2])
                self.b2.update(-self.learning_rate, gradients[3])
                self.w3.update(-self.learning_rate, gradients[4])
                self.b3.update(-self.learning_rate, gradients[5])

                net_loss += nn.as_scalar(loss)
            print(f'Epoch: {epoch}, Loss: {net_loss}')