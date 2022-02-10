#makefile

start:
	rm bin/*.*

serial:
	${GCC}/bin/g++ -march=x86-64 -O3 -fno-tree-vectorize -c -o src/pathfinder_serial.o src/pathfinder.cpp
	${GCC}/bin/g++ -march=x86-64 -O3 -fno-tree-vectorize -o bin/pathfinder_serial.exe src/pathfinder_serial.o -lm
	rm src/*.o
	#${GCC}/bin/llvm-objdump  --mattr=+m,+f,+d,+a,+c -ds  bin/pathfinder_serial.exe > bin/pathfinder_serial.dump

vector:
	${GCC}/bin/g++ -march=x86-64 -O3 -fno-tree-vectorize -mavx512f -c -o src/pathfinder.o src/pathfinder.cpp ; \
	${GCC}/bin/g++ -march=x86-64 -O3 -fno-tree-vectorize -mavx512f -o bin/pathfinder_vector.exe src/*.o -lm ; \
	rm src/*.o ; \
#	${GCC}/bin/llvm-objdump  --mattr=+m,+f,+d,+a,+c,+experimental-v -ds   bin/pathfinder_vector.exe > bin/pathfinder_vector.dump ; \

clean:
	rm bin/*.dump; \
	rm bin/*.exe;