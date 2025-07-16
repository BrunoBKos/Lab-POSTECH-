out: gemv.cu
	/usr/local/cuda/bin/nvcc gemv.cu -o gemv
sm: gemv_sm.cu
	/usr/local/cuda/bin/nvcc gemv_sm.cu -o gemv_sm
clean:
	rm -rf gemv gemv_sm
