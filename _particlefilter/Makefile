#makefile

start:
	rm bin/*.*

serial:
	${GCC}/bin/gcc -march=native -O3 -fno-tree-vectorize -c -o src/particlefilter_serial.o src/particlefilter.c
	${GCC}/bin/gcc -march=native -O3 -fno-tree-vectorize -o bin/particlefilter_serial.exe src/particlefilter_serial.o -lm
	rm src/*.o
	#${LLVM}/bin/llvm-objdump --mattr=+m,+f,+d,+a,+c -ds  bin/particlefilter_serial.exe > bin/particlefilter_serial.dump

vector:
	${GCC}/bin/gcc -march=native -mavx512f -DUSE_RISCV_VECTOR -O3 -fno-tree-vectorize -c -o src/particlefilter.o src/particlefilter.c
	${GCC}/bin/gcc -march=native -mavx512f -DUSE_RISCV_VECTOR -O3 -fno-tree-vectorize -o bin/particlefilter_vector.exe src/*.o -lm
	rm src/*.o
	#${LLVM}/bin/llvm-objdump  --mattr=+m,+f,+d,+a,+c,+experimental-v -ds   bin/particlefilter_vector.exe > bin/particlefilter_vector.dump

clean:
	rm bin/*.dump
	rm bin/*.exe
	