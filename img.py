import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

class image:
    def __init__(self, name):
        self.array = img.imread(name)

    def convolute(self, choice):
        ker = np.zeros(shape=(3, 3))
        scale = 1
        add = 0
        if choice == 'box blur 3':
            scale = 1/9
            ker = np.array([[1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.]])
        elif choice == 'box blur 5':
            scale = 1/25
            ker = np.array([[1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.]])
        elif choice == 'box blur 7':
            scale = 1/49
            ker = np.array([[1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1.]])
        elif choice == 'circle blur 3':
            scale = 1/5
            ker = np.array([[0., 1., 0.],
                            [1., 1., 1.],
                            [0., 1., 0.]])
        elif choice == 'circle blur 5':
            scale = 1/21
            ker = np.array([[0., 1., 1., 1., 0.],
                            [1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.],
                            [0., 1., 1., 1., 0.]])
        elif choice == 'circle blur 7':
            scale = 1/37
            ker = np.array([[0., 0., 1., 1., 1., 0., 0.],
                            [0., 1., 1., 1., 1., 1., 0.],
                            [1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1.],
                            [0., 1., 1., 1., 1., 1., 0.],
                            [0., 0., 1., 1., 1., 0., 0.]])
        elif choice == 'gaussian blur 3':
            scale = 1/16
            ker = np.array([[1., 2., 1.],
                            [2., 4., 2.],
                            [1., 2., 1.]])
        elif choice == 'gaussian blur 5':
            scale = 1/256
            ker = np.array([[1., 4.,  6.,  4.,  1.],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4.,  6.,  4.,  1.]])
        elif choice == 'sharpen 1 3':
            ker = np.array([[ 0., -1.,  0.],
                            [-1.,  5., -1.],
                            [ 0., -1.,  0.]])
        elif choice == 'sharpen 2 3':
            ker = np.array([[-1., -1., -1.],
                            [-1.,  9., -1.],
                            [-1., -1., -1.]])
        elif choice == 'unsharp masking 5':
            scale = -1/256
            ker = np.array([[1.,  4.,    6.,  4.,  1.],
                            [4., 16.,   24., 16.,  4.],
                            [6., 24., -476., 24.,  6.],
                            [4., 16.,   24., 16.,  4.],
                            [1.,  4.,    6.,  4.,  1.]])
        elif choice == 'edge detect 1 3':
            ker = np.array([[ 1., 0., -1.],
                            [ 0., 0.,  0.],
                            [-1., 0.,  1.]])
        elif choice == 'edge detect 2 3':
            ker = np.array([[ 0., -1.,  0.],
                            [-1.,  4., -1.],
                            [ 0., -1.,  0.]])
        elif choice == 'edge detect 3 3':
            ker = np.array([[-1., -1., -1.],
                            [-1.,  8., -1.],
                            [-1., -1., -1.]])
        elif choice == 'edge detect 3 5':
            ker = np.array([[ 0., -1., -1., -1.,  0.],
                            [-1., -1., -1., -1., -1.],
                            [-1., -1., 20., -1., -1.],
                            [-1., -1., -1., -1., -1.],
                            [ 0., -1., -1., -1.,  0.]])
        elif choice == 'left sobel 3':
            ker = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])
        elif choice == 'right sobel 3':
            ker = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        elif choice == 'top sobel 3':
            ker = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])
        elif choice == 'bottom sobel 3':
            ker = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
        elif choice == 'emboss 1 3':
            ker = np.array([[-2, -1, 0],
                            [-1, 1, 1],
                            [0, 1, 2]])
        elif choice == 'emboss 2 3':
            ker = np.array([[0, 1, 2],
                            [-1, 1, 1],
                            [-2, -1, 0]])
        elif choice == 'emboss 3 3':
            ker = np.array([[2, 1, 0],
                            [1, 1, -1],
                            [0, -1, -2]])
        elif choice == 'emboss 4 3':
            ker = np.array([[0, -1, -2],
                            [1, 1, -1],
                            [2, 1, 0]])
        elif choice == 'identity 3':
            ker = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])
        elif choice == 'negative 3':
            ker = np.array([[0, 0, 0],
                            [0, -1, 0],
                            [0, 0, 0]])
        ker *= scale
        self.handle_convolute(ker, add)

    def handle_convolute(self, ker, add):
        size = ker.shape[0]
        dim = (self.get_dim()[0] - size + 1, self.get_dim()[1] - size + 1, self.get_dim()[2])
        to_ret = np.zeros(shape=dim)
        for i in range(0, dim[0]):
            for j in range(0, dim[1]):
                for k in range(0, dim[2]):
                    to_ret[i][j][k] = self.dot_product(ker, (i+1, j+1, k))
        self.set_array(to_ret)

    def dot_product(self, ker, loc):
        sum = 0
        size = ker.shape[0]
        for i in range(size):
            for j in range(size):
                sum += ker[i][j] * self.array[loc[0] - size//2 + i][loc[1] - size//2 + j][loc[2]]
        return (sum)

    def greyscale_r(self):
        for i in range(self.get_dim()[0]):
            for j in range(self.get_dim()[1]):
                for k in range(self.get_dim()[2]):
                    self.array[i][j][k] = self.array[i][j][0]

    def greyscale_g(self):
        for i in range(self.get_dim()[0]):
            for j in range(self.get_dim()[1]):
                for k in range(self.get_dim()[2]):
                    self.array[i][j][k] = self.array[i][j][1]

    def greyscale_b(self):
        for i in range(self.get_dim()[0]):
            for j in range(self.get_dim()[1]):
                for k in range(self.get_dim()[2]):
                    self.array[i][j][k] = self.array[i][j][2]

    def greyscale_avg(self):
        for i in range(self.get_dim()[0]):
            for j in range(self.get_dim()[1]):
                avg = 0
                for l in range(self.get_dim()[2]):
                    avg += self.array[i][j][l]
                avg /= self.get_dim()[2]
                for k in range(self.get_dim()[2]):
                    self.array[i][j][k] = avg

    def print(self):
        print(self.array)

    def print1(self):
        printArray = np.zeros(shape=(self.get_dim()[0], self.get_dim()[1]))
        for i in range(self.get_dim()[0]):
            for j in range(self.get_dim()[1]):
                printArray[i][j] = self.array[i][j][1]
        print(printArray)

    def to_decimal(self):
        self.array /= 255.

    def load_img(self, name):
        self.array = img.imread(name)
        # self.to_decimal()

    def disp_img(self):
        plt.imshow(self.array)
        plt.show()

    def get_dim(self):
        # print(self.array.shape)
        return self.array.shape

    def set_array(self, input):
        self.array = input

    def override_test(self):
        self.set_array(np.zeros(shape=(6, 6, 3)))
        for i in range(6):
            for j in range(6):
                for k in range(3):
                    self.array[i][j][k] = i

    def override_test_2(self):
        self.set_array(np.zeros(shape=(6, 6, 3)))
        for i in range(6):
            for j in range(6):
                for k in range(3):
                    self.array[i][j][k] = i * j

    def override_test_3(self):
        self.set_array(np.zeros(shape=(6, 6, 3)))
        for k in range(2):
            self.array[2][2][k] = 2
            self.array[3][2][k] = 1



