import cPickle
import gzip
import os
import sys
import time
import glob

import numpy
try:
    import pylab
except ImportError:
    print (
        "pylab isn't available. If you use its functionality, it will crash."
    )
    print "It can be installed with 'pip install -q Pillow'"

from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from adam import ADAM
from lstm import LSTM

numpy.random.seed(0xbeef)
rng = RandomStreams(seed=numpy.random.randint(1 << 30))

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

class ReadHead(object):

    def __init__(self, n_hidden):
        self.wh = theano.shared(name='read_wh',
                                value=numpy.sqrt(6./(5 + n_hidden)) * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, 5))
                                .astype(theano.config.floatX))
        self.params = []

    def read(self,x,x_err,htm1):
         return T.concatenate([x,x_err],axis=1)

class WriteHead(object):

    def __init__(self, imgX, imgY,n_hidden):
        self.wh = theano.shared(name='write_wh',
                                value=numpy.sqrt(6./(imgX*imgY + n_hidden)) * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, imgX*imgY))
                                .astype(theano.config.floatX))
        self.params = [self.wh]

    def write(self,h_t):
        return T.dot(h_t,self.wh)

class RandomVariable(object):
    
    def __init__(self, srng, n):
        ''' 
        n:: dimension of input
        '''

        self.w_mean = theano.shared(name='w_mean',
                                value=numpy.sqrt(6./(2*n)) * numpy.random.uniform(-1.0, 1.0,
                                (n, n))
                                .astype(theano.config.floatX))

        self.w_var = theano.shared(name='w_var',
                                value= numpy.sqrt(6./(2*n))*numpy.random.uniform(-1.0, 1.0,
                                (n, n))
                                .astype(theano.config.floatX))
        self.n = n
        self.params = [self.w_mean,self.w_var]
        #self.params = [self.w_mean]

    def conditional_sample(self,input,epsilon):
      return T.dot(input,self.w_mean)+T.sqrt(T.exp(T.dot(input,self.w_var)))*epsilon

    def latent_loss(self,input):
      var = T.exp(T.dot(input,self.w_var))
      return 0.5*T.sum(T.dot(input,self.w_mean)**2 + var - T.log(var),axis = [0,2])

    def conditional_prob(self,outcome,condition):
      diff = (T.dot(condition,self.w_mean)-outcome)
      var = T.exp(T.dot(condition,self.w_var))
      #ignoring some multiplicative constants, disappear in derivative of log anyway
      #return (1/(1e-200+T.sqrt(T.prod(1.5*var,axis=1))))*T.exp(-0.5*T.sum(diff*(1/var)*diff,axis=1))
      return T.prod((1/(T.sqrt(2*numpy.pi*var)))*T.exp(-0.5*diff*(1/var)*diff),axis=1)

    def log_conditional_prob(self,outcome,condition):
      diff = (T.dot(condition,self.w_mean)-outcome)
      var = T.exp(T.dot(condition,self.w_var))
      #ignoring some multiplicative constants, disappear in derivative of log anyway
      #return (1/(1e-200+T.sqrt(T.prod(1.5*var,axis=1))))*T.exp(-0.5*T.sum(diff*(1/var)*diff,axis=1))
      return T.sum(T.log((1/(T.sqrt(2*numpy.pi*var)))*T.exp(-0.5*diff*(1/var)*diff)),axis=1)


class DRAW(object):

  def __init__(self, rng, input, imgX, imgY, n_hidden_enc = 100, n_hidden_dec = 100, n_steps = 1, batch_size = 1):

    #initialize parameters and 

    self.c0 = theano.shared(name='c0',
                                value=numpy.random.uniform(0.0, 1.0,
                                (imgX*imgY))
                                .astype(theano.config.floatX))

    self.rnn_enc = LSTM(n_hidden_dec+2*imgX*imgY,n_hidden_enc)
    self.rnn_dec = LSTM(n_hidden_enc,n_hidden_dec)
    self.Z = RandomVariable(rng,n_hidden_enc)
    self.readHead = ReadHead(n_hidden_enc)
    self.writeHead = WriteHead(imgX,imgY,n_hidden_dec)
    self.X = RandomVariable(rng,imgX*imgY)
    self.randSeq = rng.normal((n_steps,batch_size,n_hidden_enc))

    self.params = [self.c0] + self.readHead.params + self.rnn_enc.params + self.Z.params + self.rnn_dec.params + self.X.params + self.writeHead.params

    #turns vector into n_batches x vector_length matrix
    #concatenate operation won't broadcast so we add a 0 matrix with
    #the correct number of rows      
    def vec2Matrix(v):
      t = v.dimshuffle(['x',0])
      t = T.dot(input.dimshuffle([1,0])[0].dimshuffle([0,'x']),t)
      return v + T.zeros_like(t)

    def autoEncode(epsilon,ctm1,stm1_enc,htm1_enc,stm1_dec,htm1_dec,ztm1,x):
      x_err = x - T.nnet.sigmoid(ctm1) 
      rt = self.readHead.read(x,x_err,htm1_dec)
      [s_t_enc,h_t_enc] = self.rnn_enc.recurrence(
                T.concatenate([rt,htm1_dec],axis=1),stm1_enc,htm1_enc)
      z_t = self.Z.conditional_sample(h_t_enc,epsilon)
      [s_t_dec,h_t_dec] = self.rnn_dec.recurrence(z_t,stm1_dec,htm1_dec)
      c_t = ctm1 + self.writeHead.write(h_t_dec)
      return [c_t,s_t_enc,h_t_enc,s_t_dec,h_t_dec,ztm1+[z_t]]

    '''results, updates = theano.scan(fn = autoEncode, 
      outputs_info = [vec2Matrix(self.c0),vec2Matrix(self.rnn_enc.s0),
          vec2Matrix(self.rnn_enc.h0),vec2Matrix(self.rnn_dec.s0),
          vec2Matrix(self.rnn_dec.h0),numpy.zeros((batch_size,n_hidden_enc),
                                dtype=theano.config.floatX)], 
      sequences = self.randSeq,
      non_sequences = input, n_steps = n_steps)'''

    c_t,s_t_enc,h_t_enc,s_t_dec,h_t_dec,z_t = [vec2Matrix(self.c0),vec2Matrix(self.rnn_enc.s0),
          vec2Matrix(self.rnn_enc.h0),vec2Matrix(self.rnn_dec.s0),
          vec2Matrix(self.rnn_dec.h0),[]]

    #would like to use scan here but runs into errors with computations involving random variables
    #also takes much longer to find gradient graph

    for i in range(n_steps):
      c_t,s_t_enc,h_t_enc,s_t_dec,h_t_dec,z_t = autoEncode(self.randSeq[i],c_t,s_t_enc,h_t_enc,s_t_dec,h_t_dec,z_t,input)

    self.zT = T.stacklists(z_t)
    self.cT = c_t
    self.lossX = T.sum(-self.X.log_conditional_prob(input,self.cT))
    self.lossZ = T.sum(T.sum(self.Z.latent_loss(self.zT)) - n_steps/2)
    self.loss = (self.lossX+self.lossZ)/batch_size
    self.test = self.loss
    self.generated_x = self.X.conditional_sample(self.cT,rng.normal((batch_size,imgX*imgY)))
    self.mean = T.dot(self.cT,self.X.w_mean)
    self.var = T.exp(T.dot(self.cT,self.X.w_var))

def test_draw():

  batch_size = 100

  dataset = '../DeepLearningTutorials/data/mnist.pkl.gz'

  datasets = load_data(dataset)

  train_set_x, train_set_y = datasets[0]
  valid_set_x, valid_set_y = datasets[1]
  test_set_x, test_set_y = datasets[2]

  # compute number of minibatches for training, validation and testing
  n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
  n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
  n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

  #test_batch = test_set_x.get_value()[index * batch_size:(index + 1) * batch_size]

  train_set = train_set_x.get_value(borrow=True)
  n_train_batches = train_set.shape[0] / batch_size
  test_set = test_set_x.get_value(borrow=True)

  x = T.dmatrix('x')

  draw = DRAW(rng=rng,input=x,
    imgX=28,imgY=28,
    n_hidden_enc=28*28,n_hidden_dec=28*28,
    n_steps=2,batch_size=batch_size)

  print("... initialized graph")
  
  gparams = [T.grad(draw.loss, param) for param in draw.params]

  print("... gradient graph computed")

  adam = ADAM(draw.params,gparams)

  updates = adam.updates

  trainModel = theano.function(inputs=[x],outputs = draw.test,updates = updates)

  print("functions compiled")

  epoch = 0
  n_epochs = 10
  done_looping = False
  best_cost = 1e10
  patience = 350
  n_steps_left = patience

  while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for index in xrange(n_train_batches):

      cost = trainModel(train_set[index * batch_size:(index + 1) * batch_size])
      iter = (epoch - 1) * n_train_batches + index

      print(
          'epoch %i, minibatch %i/%i, training set cost %f' %
          (epoch, index + 1, n_train_batches, cost)
      )

      if cost < best_cost:
          print("new best")
          n_steps_left = patience
          best_cost = cost

      n_steps_left -= 1
      if n_steps_left < 0:
        done_looping = True

  doOnce = theano.function(inputs=[x],outputs = [draw.generated_x,draw.mean,draw.var])

  output = doOnce(test_set[0:100])

  pylab.gray()
  for i,img in enumerate(test_set[0:5]):
    pylab.subplot(4, 5, 1+i); pylab.axis('off'); pylab.imshow(img.reshape(28,28),vmin=0, vmax=1)

  for i,img in enumerate(output[0][0:5]):
    pylab.subplot(4, 5, 6+i); pylab.axis('off'); pylab.imshow(img.reshape(28,28),vmin=0, vmax=1)

  for i,img in enumerate(output[1][0:5]):
    pylab.subplot(4, 5, 11+i); pylab.axis('off'); pylab.imshow(img.reshape(28,28))

  for i,img in enumerate(output[2][0:5]):
    pylab.subplot(4, 5, 16+i); pylab.axis('off'); pylab.imshow(img.reshape(28,28))
    print (img.min())
    print (img.max())


  pylab.show()
  

if __name__ == '__main__':
    test_draw()

