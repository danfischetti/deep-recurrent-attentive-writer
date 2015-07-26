import numpy
import theano
import theano.tensor as T

class LSTM(object):
    ''' long short term memory neural net model '''
    def __init__(self, n_in, n_hidden):
        '''
        n_hidden :: dimension of the hidden layer
        n_in :: dimension of input
        '''

        # parameters of the model
        self.wx = theano.shared(name='wx',
                                value=numpy.sqrt(6./(n_in + n_hidden)) * numpy.random.uniform(-1.0, 1.0,
                                (n_in, n_hidden))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=numpy.sqrt(6./(n_hidden + n_hidden)) * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))
        self.wx_in = theano.shared(name='wx_in',
                                value=numpy.sqrt(6./(n_in + n_hidden)) * numpy.random.uniform(-1.0, 1.0,
                                (n_in, n_hidden))
                                .astype(theano.config.floatX))
        self.wh_in = theano.shared(name='wh_in',
                                value=numpy.sqrt(6./(n_hidden + n_hidden)) * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))
        self.ws_in = theano.shared(name='ws_in',
                                value=numpy.sqrt(6./(n_hidden + n_hidden)) * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))
        self.wx_out = theano.shared(name='wx_out',
                                value=numpy.sqrt(6./(n_in + n_hidden)) * numpy.random.uniform(-1.0, 1.0,
                                (n_in, n_hidden))
                                .astype(theano.config.floatX))
        self.wh_out = theano.shared(name='wh_out',
                                value=numpy.sqrt(6./(n_hidden + n_hidden)) * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))
        self.ws_out = theano.shared(name='ws_out',
                                value=numpy.sqrt(6./(n_hidden + n_hidden)) * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))
        self.wx_forget = theano.shared(name='wx_forget',
                                value=numpy.sqrt(6./(n_in + n_hidden)) * numpy.random.uniform(-1.0, 1.0,
                                (n_in, n_hidden))
                                .astype(theano.config.floatX))
        self.wh_forget = theano.shared(name='wh_forget',
                                value=numpy.sqrt(6./(n_hidden + n_hidden)) * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))
        self.ws_forget = theano.shared(name='ws_forget',
                                value=numpy.sqrt(6./(n_hidden + n_hidden)) * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        self.b_forget = theano.shared(name='b_forget',
                                value=1+numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        self.b_in = theano.shared(name='b_in',
                                value=0.5+numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        self.b_out = theano.shared(name='b_out',
                                value=0.5+numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        self.s0 = theano.shared(name='s0',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))

        self.params = [self.wx, self.wh, self.bh,
                       self.wx_in, self.wh_in, self.ws_in, self.b_in,
                       self.wx_out, self.wh_out, self.ws_out, self.b_out,
                       self.wx_forget, self.wh_forget, self.ws_forget, self.b_forget,
                       self.h0]

    def recurrence(self,x_t, s_tm1, y_tm1):
           in_t = T.nnet.sigmoid(T.dot(x_t, self.wx_in) + T.dot(s_tm1, self.ws_in)
                                + T.dot(y_tm1, self.wh_in) + self.b_in)
           out_t = T.nnet.sigmoid(T.dot(x_t, self.wx_out) + T.dot(s_tm1, self.ws_out)
                                + T.dot(y_tm1, self.wh_out) + self.b_out)
           forget_t = T.nnet.sigmoid(T.dot(x_t, self.wx_forget) + T.dot(s_tm1, self.ws_forget)
                                + T.dot(y_tm1, self.wh_forget) + self.b_forget)
           g_t = T.tanh(T.dot(x_t, self.wx)
                                + T.dot(y_tm1, self.wh) + self.bh)
           s_t = s_tm1*forget_t + g_t*in_t
           h_t = T.tanh(s_t)
           y_t = h_t*out_t
           return [s_t, y_t]