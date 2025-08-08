out: sssp.cu spcsr_generator.o
	/usr/local/cuda/bin/nvcc sssp.cu spcsr_generator.o -o sssp -lm
spcsr_generator.o: spcsr_generator.h spcsr_generator.cc 
	gcc spcsr_generator.o -c spcsr_generator.cc -lm
test: spcsr_generator.o test.cc spcsr_generator.h
	gcc test.cc spcsr_generator.o -o test -lm
clean:
	rm -rf sssp test *.o *.data
