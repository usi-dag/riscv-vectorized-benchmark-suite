#makefile

start:
	rm bin/*.*

serial:
	${GCC}/bin/g++ -march=native -O3 -fno-tree-vectorize -c -o src/streamcluster_serial.o src/streamcluster.cpp
	${GCC}/bin/g++ -march=native -O3 -fno-tree-vectorize -o bin/streamcluster_serial.exe src/streamcluster_serial.o -lm
	rm src/*.o
	#${GCC}/bin/llvm-objdump  --mattr=+m,+f,+d,+a,+c -ds  bin/streamcluster_serial.exe > bin/streamcluster_serial.dump

vector:
	${GCC}/bin/g++ -march=native -O3 -fno-tree-vectorize -DUSE_RISCV_VECTOR -mavx512f -c -o src/streamcluster.o src/streamcluster.cpp
	${GCC}/bin/g++ -march=native -O3 -fno-tree-vectorize -DUSE_RISCV_VECTOR -mavx512f -o bin/streamcluster_vector.exe src/*.o -lm
	rm src/*.o
	#${GCC}/bin/llvm-objdump  --mattr=+m,+f,+d,+a,+c,+experimental-v -ds   bin/streamcluster_vector.exe > bin/streamcluster_vector.dump

clean:
	rm bin/*.dump
	rm bin/*.exe