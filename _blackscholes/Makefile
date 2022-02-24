#makefile

start:
	rm bin/*.*

serial:
	${GCC}/bin/g++   -march=native -O3 -fno-tree-vectorize -c -o src/blackscholes_serial.o src/blackscholes.cpp
	${GCC}/bin/g++   -march=native -O3 -fno-tree-vectorize -o bin/blackscholes_serial.exe src/blackscholes_serial.o -lm
	rm src/*.o
#	${GCC}/bin/llvm-objdump --mattr=+m,+f,+d,+a,+c -ds  bin/blackscholes_serial.exe > bin/blackscholes_serial.dump

vector:
	${GCC}/bin/g++  -march=native -DUSE_RISCV_VECTOR -mavx512f -mveclibabi=svml -O3 -fno-tree-vectorize -c -o src/blackscholes.o src/blackscholes.cpp ; \
	${GCC}/bin/g++  -march=native -DUSE_RISCV_VECTOR -mavx512f -mveclibabi=svml -O3 -fno-tree-vectorize -o bin/blackscholes_vector.exe src/*.o -lm ; \
	rm src/*.o ; \
#	${GCC}/bin/llvm-objdump --mattr=+m,+f,+d,+a,+c,+experimental-v -ds bin/blackscholes_vector.exe > bin/blackscholes_vector.dump ;

clean:
	rm bin/*.dump; \
	rm bin/*.exe;