#makefile

start:
	mkdir bin; \
	rm bin/*.*

serial:
	for entry in src/*.cpp ; do \
		${GCC}/bin/g++ -march=native -O3 -fno-tree-vectorize -c -o  $$entry.o  $$entry ; \
	done
	${GCC}/bin/g++  -march=native -O3 -fno-tree-vectorize -o bin/swaptions_serial.exe src/*.cpp.o -lm;	\
#	${GCC}/bin/llvm-objdump  --mattr=+m,+f,+d,+a,+c -ds  bin/swaptions_serial.exe > bin/swaptions_serial.dump;	\
	rm src/*.o; \

vector:
	${GCC}/bin/g++ -march=native -mavx512f -DUSE_RISCV_VECTOR -O3 -fno-tree-vectorize -c -o src/CumNormalInv.o src/CumNormalInv.cpp;	\
	${GCC}/bin/g++ -march=native -mavx512f -DUSE_RISCV_VECTOR -O3 -fno-tree-vectorize -c -o src/MaxFunction.o src/MaxFunction.cpp;	\
	${GCC}/bin/g++ -march=native -mavx512f -DUSE_RISCV_VECTOR -O3 -fno-tree-vectorize -c -o src/RanUnif.o src/RanUnif.cpp;	\
	${GCC}/bin/g++ -march=native -mavx512f -DUSE_RISCV_VECTOR -O3 -fno-tree-vectorize -c -o src/nr_routines.o src/nr_routines.cpp;	\
	${GCC}/bin/g++ -march=native -mavx512f -DUSE_RISCV_VECTOR -O3 -fno-tree-vectorize -c -o src/icdf.o src/icdf.cpp;	\
	${GCC}/bin/g++ -march=native -mavx512f -DUSE_RISCV_VECTOR -O3 -fno-tree-vectorize -c -o src/HJM_SimPath_Forward_Blocking.o src/HJM_SimPath_Forward_Blocking.cpp;	\
	${GCC}/bin/g++ -march=native -mavx512f -DUSE_RISCV_VECTOR -O3 -fno-tree-vectorize -c -o src/HJM.o src/HJM.cpp;	\
	${GCC}/bin/g++ -march=native -mavx512f -DUSE_RISCV_VECTOR -O3 -fno-tree-vectorize -c -o src/HJM_Swaption_Blocking.o src/HJM_Swaption_Blocking.cpp;	\
	${GCC}/bin/g++ -march=native -mavx512f -DUSE_RISCV_VECTOR -O3 -fno-tree-vectorize -c -o src/HJM_Securities.o src/HJM_Securities.cpp;	\
	${GCC}/bin/g++ -march=native -mavx512f -DUSE_RISCV_VECTOR -O3 -fno-tree-vectorize -o bin/swaptions_vector.exe src/*.o -lm;	\
#	${GCC}/bin/llvm-objdump  --mattr=+m,+f,+d,+a,+c,+experimental-v -ds  bin/swaptions_vector.exe > bin/swaptions_vector.dump ; \
	rm src/*.o ;

clean:
	rm bin/*.dump
	rm bin/*.exe