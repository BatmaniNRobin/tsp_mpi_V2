todo: main
main: main.cpp
	mpicxx -pthread -o main main.cpp -fopenmp
clean:
	rm main