#makefile

start:
	rm bin/*.*
	
serial:
	${GCC}/bin/g++ -march=native  -O3 -fno-tree-vectorize -c -o src/jacobi-2d.o src/jacobi-2d.cpp
	${GCC}/bin/g++ -march=native  -O3 -fno-tree-vectorize -o bin/jacobi-2d_serial.exe src/*.o -lm
	rm src/*.o
	#${LLVM}/bin/llvm-objdump  --mattr=+m,+f,+d,+a,+c -ds  bin/jacobi-2d_serial.exe > bin/jacobi-2d_serial.dump

vector:
	${GCC}/bin/g++ -DUSE_RISCV_VECTOR -march=native -mavx512f -O3 -fno-tree-vectorize -c -o src/jacobi-2d.o src/jacobi-2d.cpp
	${GCC}/bin/g++ -DUSE_RISCV_VECTOR -march=native -mavx512f -O3 -fno-tree-vectorize -o bin/jacobi-2d_vector.exe src/*.o -lm
	rm src/*.o
	#${LLVM}/bin/llvm-objdump  --mattr=+m,+f,+d,+a,+c,+experimental-v -ds   bin/jacobi-2d_vector.exe > bin/jacobi-2d_vector.dump

clean:
	rm bin/*.dump
	rm bin/*.exe