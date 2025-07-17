out: softmax.cu
	/usr/local/cuda/bin/nvcc softmax.cu -o softmax -lm
sm: softmax_sm.cu
	/usr/local/cuda/bin/nvcc softmax_sm.cu -o softmax_sm -lm
clean:
	rm -rf softmax softmax_sm
