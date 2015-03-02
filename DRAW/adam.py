import numpy

import theano
import theano.tensor as T

class ADAM (object):

	def __init__(self,params,gparams,alpha = 0.0002,
		beta1 = 0.1,beta2 = 0.001,epsilon = 1e-8, l = 1e-8):

		self.m = [theano.shared(name = 'm',
			value = numpy.zeros(param.get_value().shape,dtype=theano.config.floatX)) for param in params]
		self.v = [theano.shared(name = 'v',
			value = numpy.zeros(param.get_value().shape,dtype=theano.config.floatX)) for param in params]
		self.t = theano.shared(name = 't',value = 1)

		self.updates = [(self.t,self.t+1)]
		for param,gparam,m,v in zip(params,gparams,self.m,self.v):
			b1_t = 1-(1-beta1)*(l**(self.t-1))
			m_t = b1_t*gparam + (1-b1_t)*m
			self.updates.append((m,m_t))
			v_t = beta2*(gparam**2)+(1-beta2)*v
			self.updates.append((v,v_t))
			m_t_bias = m_t/(1-(1-beta1)**self.t)	
			v_t_bias = v_t/(1-(1-beta2)**self.t)
			self.updates.append((param,param - alpha*m_t_bias/(T.sqrt(v_t_bias)+epsilon)))		

