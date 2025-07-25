test_weight: weight.o weight.h
	/usr/local/cuda/bin/nvcc test_weight.cc weight.o -o test
test_bias: bias.o bias.h
	/usr/local/cuda/bin/nvcc test_bias.cc bias.o -o test
test_relu: relu.o relu.h
	/usr/local/cuda/bin/nvcc test_relu.cc relu.o -o test
test_sigmoid: sigmoid.o sigmoid.h
	/usr/local/cuda/bin/nvcc test_sigmoid.cc sigmoid.o -o test -lm
test_tanh: tanh.o tanh.h
	/usr/local/cuda/bin/nvcc test_tanh.cc tanh.o -o test -lm
test_gelu: gelu.o gelu.h
	/usr/local/cuda/bin/nvcc test_gelu.cc gelu.o -o test -lm
weight.o: weight.cu weight.h
	/usr/local/cuda/bin/nvcc weight.o -c weight.cu
bias.o: bias.cu bias.h
	/usr/local/cuda/bin/nvcc bias.o -c bias.cu
relu.o: relu.cu relu.h
	/usr/local/cuda/bin/nvcc relu.o -c relu.cu
sigmoid.o: sigmoid.cu sigmoid.h
	/usr/local/cuda/bin/nvcc sigmoid.o -c sigmoid.cu
tanh.o: tanh.cu tanh.h
	/usr/local/cuda/bin/nvcc tanh.o -c tanh.cu
gelu.o: gelu.cu gelu.h
	/usr/local/cuda/bin/nvcc gelu.o -c gelu.cu
clean:
	rm -rf *.o test