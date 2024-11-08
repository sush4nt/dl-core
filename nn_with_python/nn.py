import sys
import logging
import numpy as np
from numpy import float64, ndarray
from typing import List, Tuple, Union

logger = logging.getLogger(__name__) # logger object for logging
class NeuralNetwork:
    def __init__(self, 
        layer_dimensions: list = [], 
        activations: list = []) -> None:
        '''
        > Initializes networks's weights and other useful variables.
        
        > Parameters: <br>
        `layer_dimensions`: specify the dimensions for input + hidden + output layer
        `activations`: to store the activation for each hidden + output layer
        
        > Attributes: <br>
        -`parameters` contains weights of the layer in form {'Wi':[],'bi':[]} <br>
        -`cache` contains intermediate results as [[A[i-1],Wi,bi],[Zi]], where i
             is layer number.
        -`activations` contains the names of activation function used for that layer <br>
        -`cost_function`  contains the name of cost function to be used <br>
        -`lamb` contains the regularization hyper-parameter <br>
        -`grads` contains the gradients calculated during back-prop in form {'dA(i-1)':[],'dWi':[],'dbi':[]} <br>
        -`layer_type` contains the info about the type of layer( fc, conv etc) <br>
        
        > Results: <br>
        return: None
        '''
        self.parameters = {}
        self.cache = []
        self.activations = activations
        self.cost_function = ''
        self.lamb = 0
        self.grads = {}
        self.layer_type = ['']
        self.hyperparam = {}
        self.initialize_parameters(layer_dimensions)
        self.check_activations()
        
    def __repr__(self):
        return """Class containing functionality to build and train neural networks. \n
                  Contains all activation and loss functions some other utility function."""
    
    def __str__(self) -> str:
        '''
        Returns: the network architecture and connectivity
        '''
        logger.info('-----Network Architecture-----')
        net_string = ""
        for params in range(int(len(self.parameters)/2)):
            weight = self.parameters['W'+str(params+1)]
            if self.activations[params] != None:
                print(f"----->> Layer {params+1}: ( {str(weight.shape[1])},"
                      f"{str(weight.shape[0])} ) |  activation: {self.activations[params]} "
                      f"| No of parameters: {weight.shape[1]*weight.shape[0]}")
                sys.stdout.flush()
        return net_string

    def initialize_parameters(self, 
        layer_dimensions: list) -> None:
        '''
        `Xavier initialization` of weights of a network described by given layer
        dimensions.
        
        Parameters:
        layer_dimensions: dimensions to layers of the network
        
        Results:
        return: None
        '''
        logger.info('-----Initializing Weights-----')
        num_layers = int(len(self.parameters)/2)

        for i in range(1, len(layer_dimensions)):
            sq_root = np.sqrt( 2 / (layer_dimensions[i - 1]+layer_dimensions[i]) )
            m_shape = np.random.randn(layer_dimensions[i], layer_dimensions[i - 1])
            
            self.parameters["W" + str(num_layers+i)] = ( sq_root * m_shape )
            self.parameters["b" + str(i+num_layers)] = np.zeros((layer_dimensions[i], 1))
            self.layer_type.append('fc')

    def add_fcn(self,
        dims: list,
        activations: list) -> None:
        '''
        Add fully connected layers in between the network
        
        Parameters:
        dims: list describing dimensions of fully connected networks
        activations: activations of each layer
        '''
        self.initialize_parameters(dims)
        for i in activations:
            self.activations.append(i)

    def check_activations(self) -> None:
        '''
        Checks if activations for all layers are present. Adds 'None' if no activations are provided for a particular layer.
        
        Results:
        returns: None
        '''
        logger.info('-----Checking Activations-----')
        num_layers = int(len(self.parameters)/2)
        while len(self.activations) < num_layers :
            self.activations.append(None)
        

    @staticmethod
    def __linear_forward(A_prev: np.array, 
        W: np.array, 
        b: np.array):
        '''
        Linear forward to the current layer using previous activations.
        
        Parameters:
        param A_prev: previous Layer's activation
        param W: weights for current layer
        param b: biases for current layer
        
        Results:
        Z: current calculated layer
        linear_cache: linear cache ( previous act i/p, weights ) 
        '''
        Z = W.dot(A_prev) + b
        linear_cache = [A_prev, W, b]
        return Z, linear_cache

    def __activate(self, 
        Z, 
        n_layer=1):
        """
        Activate the given layer(Z) using the activation function specified by
            'type'.

        Note: This function treats 1 as starting index!
              First layer's index is 1.

        Parameters:
        param Z: layer to activate
        param n_layer: layer's index
        
        Results:
        act: activated layer
        act_cache: activation cache
        """
        act_cache = [Z]
        act = None
        if (self.activations[n_layer - 1]) == None:
            act = Z
        elif (self.activations[n_layer - 1]).lower() == "relu":
            act = Z * (Z > 0)
        elif (self.activations[n_layer - 1]).lower() == "tanh":
            act = np.tanh(Z)
        elif (self.activations[n_layer - 1]).lower() == "sigmoid":
            act = 1 / (1 + np.exp(-Z))
        elif (self.activations[n_layer - 1]).lower() == "softmax":
            act = np.exp(Z-np.max(Z))
            act = act/(act.sum(axis=0)+1e-10)
        return act, act_cache

    def forward(self, 
        net_input: np.array) -> np.array:
        """
        To forward propagate the entire neural network.

        ParametersL
        net_input: contains the input to the network
        :return: output of the network
        """
        self.cache = [] 
        A = net_input
        for i in range(1, int(len(self.parameters) / 2)):
            W = self.parameters["W" + str(i)]
            b = self.parameters["b" + str(i)]
            Z = linear_cache = None
            if self.layer_type[i] == 'fc':
                Z, linear_cache = self.__linear_forward(A, W, b)
            elif self.layer_type[i] == 'conv':
                hyperparam = self.hyperparam[i]
                Z , linear_cache = self.conv_forward(A,W,b,hyperparam)

                #flatten the output if the next layer is fully connected
            A, act_cache = self.__activate(Z, i)
            if  self.layer_type[i]=='conv':
                if  self.layer_type[i+1] == 'fc':
                    A = A.reshape((A.shape[1]*A.shape[2]*A.shape[3],A.shape[0]))
 

            self.cache.append([linear_cache, act_cache])

        # For Last Layer
        W = self.parameters["W" + str(int(len(self.parameters) / 2))]
        b = self.parameters["b" + str(int(len(self.parameters) / 2))]
        Z, linear_cache = self.__linear_forward(A, W, b)
        if len(self.activations) == len(self.parameters) / 2:
            A, act_cache = self.__activate(Z, len(self.activations))
            self.cache.append([linear_cache, act_cache])
        else:
            A = Z
            self.cache.append([linear_cache, [None]])
        return A
    '''
    !!!!Only works for fully connected networks.!!!!!

    def forward_upto(self, net_input, layer_num):
        """
        Calculates forward prop upto layer_num.

        :param net_input: Contains the input to the Network
        :param layer_num: Layer up to which forward prop is to be calculated
        :return: Activations of layer layer_num
        """
        if layer_num == int(len(self.parameters) / 2):
            return self.forward(net_input)
        else:
            A = net_input
            for i in range(1, layer_num):
                W = self.parameters["W" + str(i)]
                b = self.parameters["b" + str(i)]
                Z, linear_cache = self.__linear_forward(A, W, b)

                A, act_cache = self.__activate(Z, i)
                self.cache.append([linear_cache, act_cache])
            return A
    ''' 

    def MSELoss(self, 
        prediction: np.array,
        mappings: np.array) -> float:
        '''
        Calculates the `Mean Squared Error` with regularization cost(if provided) between 
            output of the network and the real mappings of a function.
        Changes `cost_function` to appropriate value

        Parameters:
        prediction: output of the neural net
        mappings: real outputs of a function
        
        Results:
        return: mean squared error b/w output and mappings
        '''
        self.cost_function = 'MSELoss'
        loss = np.square(prediction-mappings).mean()/2
        regularization_cost = 0
        if self.lamb != 0:
            for params in range(len(self.cache)):  
                regularization_cost = regularization_cost + np.sum( np.square( self.parameters['W'+str(params+1)] ) ) #L2 regn
        regularization_cost = (self.lamb/(2*prediction.shape[1]))*regularization_cost
        return loss + regularization_cost

    def CrossEntropyLoss(self,
        prediction: np.array,
        mappings: np.array) -> float:
        '''
        Calculates the cross entropy loss between output of the network and the real mappings of a function
        Changes `cost_function` to appropriate value
        
        Parameters:
        prediction: output of the neural net
        mappings: real outputs of a function
        
        Results:
        return: mean squared error b/w output and mappings
        '''
        epsilon = 1e-8
        self.cost_function = 'CrossEntropyLoss'
        ce_loss = np.sum( mappings*np.log(prediction+epsilon) + (1-mappings)*np.log(1-prediction+epsilon) )
        loss = -(1/prediction.shape[1])*ce_loss
        regularization_cost = 0
        if self.lamb != 0:
            for params in range(len(self.cache)):
                regularization_cost = regularization_cost + np.sum(np.square(self.parameters['W'+str(params+1)]))
        regularization_cost = (self.lamb/(2*prediction.shape[1]))*regularization_cost
        return loss + regularization_cost
    
    def output_backward(self,
        prediction: np.array,
        mapping: np.array) -> np.array:
        '''
        Calculates the derivative of the output layer(dA)
        
        Parameters:
        prediction: output of neural network
        mapping: correct output of the function
        cost_type: type of cost function used
        
        Results:
        return: derivative of output layer, dA  
        '''
        dA = None
        cost = self.cost_function
        if cost.lower() == 'crossentropyloss':
            # -1 * [ y/a + 1-y/1-a ]
            dA =  -(np.divide(mapping, prediction+1e-10) - np.divide(1 - mapping, 1 - prediction+1e-10))
        elif cost.lower() == 'mseloss':   
            dA =  (prediction-mapping)
        return dA
    
    def __deactivate(self,
        dA: np.array,
        n_layer: int) -> Union[np.array, int]:
        '''
        Calculates the derivate of dA by deactivating the layer

        Parameters:
        dA: Activated derivative of the layer
        n_layer: Layer number to be deactivated
        
        Results
        return: deact=> derivative of activation 
        '''
        act_cache = self.cache[n_layer-1][1]
        dZ = act_cache[0]
        deact = None
        if self.activations[n_layer - 1] == None:
            deact = 1
        elif (self.activations[n_layer - 1]).lower() == "relu":
            deact = 1* (dZ>0)
        elif (self.activations[n_layer - 1]).lower() == "tanh":
            deact = 1- np.square(dA)
        elif (self.activations[n_layer - 1]).lower() == "sigmoid" or (self.activations[n_layer - 1]).lower()=='softmax':
            s = 1/(1+np.exp(-dZ+1e-10))
            deact = s*(1-s)
        return deact
    
    def __linear_backward(self,
        dA: np.array,
        n_layer: int) -> Tuple[np.array, np.array, np.array]:
        '''
        Calculates linear backward propragation for layer denoted by n_layer

        Parameters:
        dA: Derivative of cost w.r.t this layer
        n_layer: layer number
        
        Results:
        return : dZ,dW,db,dA_prev
        '''
        batch_size = dA.shape[1]
        current_cache = self.cache[n_layer-1]
        linear_cache = current_cache[0]
        A_prev,W,b = linear_cache

        dZ = dA*self.__deactivate(dA,n_layer)
        dW = (1/batch_size)*dZ.dot(A_prev.T) + (self.lamb/batch_size)*self.parameters['W'+str(n_layer)]
        db = (1/batch_size)*np.sum(dZ,keepdims=True,axis=1)
        dA_prev = W.T.dot(dZ)

        assert dA_prev.shape == A_prev.shape, "derivatives' shape dont match: A_prev"
        assert dW.shape == W.shape, "derivatives' shape dont match: dW"
        assert db.shape == b.shape, "derivatives' shape dont match: db"
        
        return dW,db,dA_prev

    def backward(self,
        prediction: np.array,
        mappings: np.array) -> None:
        '''
        Backward propagates through the network and stores useful calculations
        
        Parameters:
        prediction: Output of neural net
        mapping: Correct output of the function
        
        Results:
        return : None
        '''
        layer_num = len(self.cache)
        doutput = self.output_backward(prediction, mappings)
        self.grads['dW'+str(layer_num)], self.grads['db'+str(layer_num)], self.grads['dA'+str(layer_num-1)] = self.__linear_backward(doutput,layer_num)
        temp = self.layer_type
        self.layer_type = self.layer_type[1:]
        
        for l in reversed(range(layer_num-1)):
            dW,db,dA_prev = None,None,None
            if self.layer_type[l] == 'fc':
                dW,db,dA_prev = self.__linear_backward(self.grads['dA'+str(l+1)],l+1)
            elif self.layer_type[l] == 'conv':
                dW,db,dA_prev = self.conv_backward((self.cache[l][1][0]),self.cache[l][0])
            self.grads['dW'+str(l+1)] = dW
            self.grads['db'+str(l+1)] = db
            self.grads['dA'+str(l)] = dA_prev
        
        self.layer_type = temp
    
    # @staticmethod
    # def zero_pad(
    #     imgData: np.array,
    #     pad: int) -> np.array:
    #     '''
    #     Provides zero padding to the multi channel image data provided
        
    #     Parameters:
    #     imgData: image data to pad
    #     pad: amount of padding per layer

    #     Results:
    #     return : image with desired padding
    #     '''
    #     X = np.pad(imgData,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values = 0)
    #     return X

    # def conv2d(self,
    #     in_planes: int,
    #     out_planes: int,
    #     kernel_size: int,
    #     activation: str,
    #     stride: int = 1,
    #     padding: int = 0) -> None:
    #     '''
    #     Add paramters for this layer in the parameters list; initialize the weights for the network
        
    #     Parameters:
    #     in_planes:
    #     out_planes:
    #     kernel_size: 
    #     activation:
    #     stride:
    #     padding:
        
    #     Results:
    #     return : None
    #     '''
    #     num_layers = int(len(self.parameters)/2)
    #     self.parameters['W'+str(num_layers+1)] = np.random.randn(kernel_size,kernel_size,in_planes,out_planes)
    #     self.parameters['b'+str(num_layers+1)] = np.random.randn(1,1,1,out_planes)
    #     self.activations.append(activation)
    #     self.layer_type.append('conv')
    #     self.hyperparam[num_layers+1] = list((stride,padding))

    # def conv_single(self,
    #     a_prev_slice: np.array,
    #     W: np.array,
    #     b: np.array) -> float:
    #     '''
    #     Apply convolution using W and b as filter on the activation slice of the previous layer

    #     Parameters:
    #     a_prev_slice: a slice of previous activated layer
    #     W           : Filter
    #     b           : bais
        
    #     Results:
    #     return Z: scalar value resultant of the convolution
    #     '''
    #     Z  = np.multiply(a_prev_slice,W)
    #     Z = np.sum(Z)
    #     Z = Z + float(b) #to convert the value to float from matrix type
    #     return Z
    
    # def conv_forward(self,
    #     A_prev: np.array,
    #     W: np.array,
    #     b: np.array,
    #     hyper_param: list) -> Tuple[np.array, Tuple[np.array, np.array, np.array, list]]:
    #     '''
    #     Implements forward pass of convolutional layer.
        
    #     Parameters:
    #     A_prev:activations of previous layer
    #     W: Filter
    #     b: bias
    #     hyper_param  : list of hyperparameters, stride and padding

    #     Results:
    #     return: Z, cache
    #     '''
    #     m,h_prev,w_prev,nc_prev = A_prev.shape
    #     f,f,nc_prev,nc = W.shape
    #     stride,pad = hyper_param
    #     #comupte the dimensions of the result using convolution formula => w/h = (w/h(prev) -f +2*pad)/stride +1
    #     n_h = int(np.floor((h_prev-f+2*pad)/stride)) +1
    #     n_w = int(np.floor((w_prev-f+2*pad)/stride)) +1
        
    #     Z = np.zeros((m,n_h,n_w,nc))
    #     A_prev_pad = self.zero_pad(A_prev,pad)
    #     for i in range(m):
    #         prev_pad = A_prev_pad[i]

    #         for h in range(n_h):
    #             for w in range(n_w):
    #                 for c in range(nc):
    #                     vertstart = h*stride
    #                     vertend = vertstart + f
    #                     horstart = w*stride
    #                     horend = horstart+f 

    #                     prev_slice = prev_pad[vertstart:vertend,horstart:horend,:]
    #                     Z[i,h,w,c] = self.conv_single(prev_slice,W[:,:,:,c],b[:,:,:,c])
        
    #     cache = (A_prev,W,b,hyper_param)

    #     return Z,cache

    # def pool_forward(self,
    #     A_prev: np.array,
    #     f: int,
    #     stride: int,
    #     type='max'):
    #     '''
    #     To enable max and average pooling during the forward pass
        
    #     Parameters:
    #     A_prev: Activation of previous layer
    #     f: filter size
    #     stride: size of each stride
    #     type: type of pooling, max or average
        
    #     Results:
    #     returns A,cache:
    #     '''
    #     #Calculate the resultant dimensions:
    #     n_h = int(1 + (A_prev.shape[1] - f) / stride)
    #     n_w = int(1 + (A_prev.shape[2] - f) / stride)
    #     n_c = A_prev.shape[3]

    #     A = np.zeros((A_prev.shape[0],n_h,n_w,n_c))

    #     for i in range(A.shape[0]):
    #         for h in range(n_h):
    #             for w in range(n_w):
    #                 for c in range(n_c):
    #                     vertstart = h*stride
    #                     vertend = vertstart + f
    #                     horstart = w*stride
    #                     horend = horstart+f 

    #                     a_prev_slice = A_prev[i,vertstart:vertend,horstart:horend,c]

    #                     if type == 'max':
    #                         A[i,h,w,c] = np.max(a_prev_slice)
    #                     elif type == 'avg':
    #                         A[i,h,w,c] = np.mean(a_prev_slice)
        
    #     cache = (A_prev,[f,stride],type)
    #     return A,cache




    # def conv_backward(self,
    #     dZ: ndarray, 
    #     cache: Tuple[np.array, np.array, np.array, List[int]]) -> Tuple[np.array, np.array, np.array]:
    #     '''
    #     Implement the backward propagation for a convolution function
        
    #     Parameters:
    #     dZ: adient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    #     cache: cache of values needed for the conv_backward(), output of conv_forward()
        
    #     Returns:
    #     dA_prev: gradient of the cost with respect to the input of the conv layer (A_prev),
    #                numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    #     dW: gradient of the cost with respect to the weights of the conv layer (W)
    #           numpy array of shape (f, f, n_C_prev, n_C)
    #     db: gradient of the cost with respect to the biases of the conv layer (b)
    #           numpy array of shape (1, 1, 1, n_C)
    #     '''
        

    #     (A_prev, W, b, hparameters) = cache
        
    #     (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
    #     (f, f, n_C_prev, n_C) = W.shape
        
    #     stride = hparameters[0]
    #     pad = hparameters[1]
        
    #     (m, n_H, n_W, n_C) = dZ.shape
        
    #     dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
    #     dW = np.zeros((f, f, n_C_prev, n_C))
    #     db = np.zeros((1, 1, 1, n_C))

    #     A_prev_pad = self.zero_pad(A_prev, pad)
    #     dA_prev_pad = self.zero_pad(dA_prev, pad)
        
    #     for i in range(m):                      
            
    #         a_prev_pad = A_prev_pad[i]
    #         da_prev_pad = dA_prev_pad[i]
            
    #         for h in range(n_H):                  
    #             for w in range(n_W):               
    #                 for c in range(n_C):           
                        
    #                     vert_start = h * stride

    #                     vert_end = vert_start + f
    #                     horiz_start = w * stride

    #                     horiz_end = horiz_start + f
                        
    #                     a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

    #                     da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
    #                     dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
    #                     db[:,:,:,c] += dZ[i, h, w, c]
                        
    #         dA_prev[i, :, :, :] =  da_prev_pad if pad == 0 else da_prev_pad[pad:-pad,pad:-pad,:]
        
    #     assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
        
    #     return dA_prev, dW, db


 
    
    # def create_mask(self,X):
    #     '''
    #     Creates mask of from a slice which sets max element index to 1 and others to 0

    #     :param X: original matrix
    #     :return :mask
    #     '''
    #     mask = (X==np.max(X))

    #     return mask

    # def average_back(self,X,shape):
    #     '''
    #     Computes backward pass for average pooling layer

    #     :param X: average pooled layer
    #     :param shape: shape of the original matrix
    #     '''
    #     h,w = shape
    #     X = X/(h*w)
    #     return np.ones(shape)*X

    # def pool_backward(self,
    #     dA, 
    #     cache, 
    #     mode = "max"):
    #     """
    #     Implements the backward pass of the pooling layer
        
    #     :param dA: gradient of cost with respect to the output of the pooling layer, same shape as A
    #     :param cache: cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    #     :param mode:the pooling mode you would like to use, defined as a string ("max" or "average")
        
    #     Returns:
    #     dA_prev  gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    #     """
        
    #     (A_prev, (stride,f),type) = cache
        
    #     m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    #     m, n_H, n_W, n_C = dA.shape
        
    #     dA_prev = np.zeros(A_prev.shape)
        
    #     for i in range(m):                       
    #         a_prev = A_prev[i]
    #         for h in range(n_H):                   
    #             for w in range(n_W):               
    #                 for c in range(n_C):           
                        
    #                     vert_start = h*stride
    #                     vert_end = vert_start + f
    #                     horiz_start = w*stride
    #                     horiz_end = horiz_start + f
                        
                        
    #                     if type == "max":
                        
    #                         a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
    #                         mask = self.create_mask(a_prev_slice)
    #                         dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
    #                     elif mode == "average":
    #                         da = dA[i, h, w, c]
    #                         shape = (f, f)
    #                         dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += self.average_back(da, shape)
                        
    
    
    #     return dA_prev
