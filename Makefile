CC=mpicc

override CFLAGS += -O3 -ffast-math -march=native
#override CFLAGS=-g

polybench.o: utilities/polybench.c utilities/polybench.h
	$(CC) $(CFLAGS) $(PB_FLAGS) -c -I utilities utilities/polybench.c -o $@ 

lu.o: linear-algebra/solvers/lu/lu.c linear-algebra/solvers/lu/lu.h
	$(CC) $(CFLAGS) $(PB_FLAGS) -c -I utilities -I linear-algebra/solvers/lu linear-algebra/solvers/lu/lu.c -o $@

ludcmp.o: linear-algebra/solvers/ludcmp/ludcmp.c linear-algebra/solvers/ludcmp/ludcmp.h
	$(CC) $(CFLAGS) $(PB_FLAGS) -c -I utilities -I linear-algebra/solvers/lu -I linear-algebra/solvers/ludcmp linear-algebra/solvers/ludcmp/ludcmp.c -DREUSE_LU_KERNEL -o $@

lu: polybench.o lu.o
	$(CC) $(CFLAGS) lu.o polybench.o -o lu

ludcmp: polybench.o lu.o ludcmp.o
	$(CC) $(CFLAGS) lu.o polybench.o -o ludcmp

clean:
	rm -f *.o lu ludcmp
