import cPickle
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
from logistic_sgd import load_data

numpy.random.seed(0xbeef)
rng = RandomStreams(seed=numpy.random.randint(1 << 30))

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
    
    def __init__(self, srng, n, sigmoid_mean=False):
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
        self.sigmoid_mean = sigmoid_mean
        if(self.sigmoid_mean):
          self.params = [self.w_var]
        else:
          self.params = [self.w_mean,self.w_var]
        #self.params = [self.w_mean]

    def conditional_sample(self,input,epsilon):
      if(self.sigmoid_mean):
        return T.nnet.sigmoid(input)+T.sqrt(T.exp(T.dot(input,self.w_var)))*epsilon
      else:
        return T.dot(input,self.w_mean)+T.sqrt(T.exp(T.dot(input,self.w_var)))*epsilon

    def latent_loss(self,input):
      var = T.exp(T.dot(input,self.w_var))
      return 0.5*T.sum(T.dot(input,self.w_mean)**2 + var - T.log(var),axis = [0,2])

    def log_conditional_prob(self,outcome,condition):
      if(self.sigmoid_mean):
        diff = T.nnet.sigmoid(condition)-outcome
      else:
        diff = (T.dot(condition,self.w_mean)-outcome)
      var = .005+T.exp(T.dot(condition,self.w_var))
      return T.sum(T.log((1/(T.sqrt(2*numpy.pi*var)))*(1e-10+T.exp(-0.5*diff*(1/var)*diff))),axis=1)


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
    self.X = RandomVariable(rng,imgX*imgY,sigmoid_mean=True)
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
    #diff = (T.dot(self.cT,self.X.w_mean)-input)
    #var = T.exp(T.dot(self.cT,self.X.w_var))
    self.test = self.loss
    self.generated_x = self.X.conditional_sample(self.cT,rng.normal((batch_size,imgX*imgY)))
    self.mean = T.dot(self.cT,self.X.w_mean)
    self.var = T.exp(T.dot(self.cT,self.X.w_var))

def test_draw():

  batch_size = 100

  dataset = '../data/mnist.pkl.gz'

  datasets = load_data(dataset)

  train_set_x, train_set_y = datasets[0]
  valid_set_x, valid_set_y = datasets[1]
  test_set_x, test_set_y = datasets[2]

  # compute number of minibatches for training, validation and testing
  n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
  n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
  n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

  #train_set = train_set_x.get_value(borrow=True)
  #n_train_batches = train_set.shape[0] / batch_size
  test_set = test_set_x.get_value(borrow=True)

  index = T.lscalar('index')
  x = T.matrix('x')

  draw = DRAW(rng=rng,input=x,
    imgX=28,imgY=28,
    n_hidden_enc=28*28,n_hidden_dec=28*28,
    n_steps=5,batch_size=batch_size)

  print("... initialized graph")
  
  gparams = [T.grad(draw.loss, param) for param in draw.params]

  print("... gradient graph computed")

  adam = ADAM(draw.params,gparams)

  updates = adam.updates

  trainModel = theano.function(inputs=[index],outputs = draw.test,
    updates = updates,givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]})

  print("...trainModel compiled")

  validateModel = theano.function(inputs=[index],outputs = draw.loss,
    givens={x: valid_set_x[index * batch_size: (index + 1) * batch_size]})

  print("...validateModel compiled")

  epoch = 0
  min_epochs = 5
  max_epochs = 30
  done_looping = False
  best_loss = 1e10
  patience = 10
  n_steps_left = patience

  while (epoch < max_epochs) and (not done_looping):
    epoch = epoch + 1
    for index in xrange(n_train_batches):

      cost = trainModel(index)
      iter = (epoch - 1) * n_train_batches + index

      valid_loss = validateModel(index%n_valid_batches)

      print(
          'epoch %i, minibatch %i/%i, training set cost %f' %
          (epoch, index + 1, n_train_batches, cost)
      )
      #print(cost)

      if(index%(n_train_batches/10) == 0):
        valid_losses = [validateModel(i) for i in xrange(n_valid_batches)]
        avg_valid_loss = numpy.mean(valid_losses)

        print("validation loss %f" % (avg_valid_loss))

        if avg_valid_loss < best_loss:
            print("new best")
            n_steps_left = patience
            best_loss = avg_valid_loss

        n_steps_left -= 1
        if n_steps_left < 0 and epoch > min_epochs:
          done_looping = True

  doOnce = theano.function(inputs=[x],outputs = [draw.generated_x,draw.mean,draw.var])

  output = doOnce(test_set[0:100])

  pylab.gray()
  for i,img in enumerate(test_set[0:100]):
    pylab.subplot(10, 20, 1+i); pylab.axis('off'); pylab.imshow(img.reshape(28,28),vmin=0, vmax=1)

  for i,img in enumerate(output[0]):
    pylab.subplot(10, 20, 101+i); pylab.axis('off'); pylab.imshow(img.reshape(28,28),vmin=0, vmax=1)

  pylab.show()

  param_file = open('../data/params.pk','wb')

  for param in draw.params:
    cPickle.dump(param.get_value(borrow=True),param_file,-1)

  param_file.close()
  

if __name__ == '__main__':
    test_draw()

