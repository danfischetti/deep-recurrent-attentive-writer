import theano
import cPickle
from draw import DRAW
import pylab
from PIL import Image

def test_gen():

	draw = DRAW(
    imgX=28,imgY=28,
    n_hidden_enc=256,n_hidden_dec=256,
    n_z=100,batch_size=100,
    n_steps=1)

	f = open('../data/params.pk','rb')


	for param in draw.params:
		param.set_value(cPickle.load(f))

	generate = theano.function(inputs = [],outputs = draw.generated_x2)

	pylab.gray()

	for i,img in enumerate(generate()):
		pylab.subplot(10, 10, 1+i); pylab.axis('off'); pylab.imshow(img.reshape(28,28),vmin=0, vmax=1)

	pylab.show()

if __name__ == '__main__':
    test_gen()