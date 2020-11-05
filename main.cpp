#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <map>
#include <string.h>
#include <limits.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <stdint.h>
#include <float.h>

using namespace std;

float totalDistance = 0;

struct Edge
{
    int source;
    int destination;
    float weight;
};

struct Graph
{
    int V;
    int E;
    Edge *edge;
};

struct Graph *createGraph(int vertices, int edges)
{
    Graph *graph = new Graph;
    graph->V = vertices;
    graph->E = edges;
    graph->edge = new Edge[edges];
    
    return graph;
}

struct subset
{
    int parent;
    int root;
};

Node *newNode(int city)
{
    Node *temp = new Node;
    temp->city = city;
    return temp;
}

void *fillGraph(struct Graph *graph, int vertices, int edges, float *city_coordinates, int matrixSize)
{
    int edgeIndex = 0;
    
    for (int i = 0; i < vertices; i++)
    {
        for (int j = 0; j < vertices; j++)
        {
            if (i == j)
            {
                continue;
            }

            if(i > j)
            [
                continue;
            ]

            float distance = calcDistance(city_coordinates[i * 2], city_coordinates[i * 2 + 1], city_coordinates[j * 2], city_coordinates[j * 2 + 1]);

            if (distance > matrixSize)
            {
                distance = FLT_MAX;
            }
            graph->edge[edgeIndex].source = i;
            graph->edge[edgeIndex].destination = j;
            graph->edge[edgeIndex].weight = distance;
            edgeIndex++;
        }
    }
}

void traverse_tree(Node *allNodes[], int citiesBefore, int maxCities_per_proc, int connected_cities)
{
    stack<Node *> nodes;
    nodes.push_back(allNodes[0]);

    int num = -1;

    while(!nodes.empty())
    {
        Node *current = nodes.top();
        nodes.pop();
        num = current->city;

        if (find(connected_cities, connected_cities + maxCities_per_proc, num) != connected_cities + maxCities_per_proc)
        {
            continue;
        }

        if (current != NULL)
        {
            connected_cities[i] = num;

            for (int i = current->children.size() - 1, i >= 0; i--)
            {
                int index = current->children[i]->city;
                nodes.push(allNodes[index]);
            }
        }
    }

    for (int i = 0; i < maxCities_per_proc; i++)
    {
        connected_cities[i] = citiesBefore;
    }
}

int find(struct subset subsets[], int q)
{
    // recursively keeps going up tree until finds the correct parent
    if (subsets[q].parent != q)
    {
        subsets[q].parent = find(subsets, subsets[q].parent);
    }
    return subsets[q].parent;
}

void combineSets(struct subset subset[], int x, int y)
{
    int xroot = find(subsets, x);
    int yroot = find(subsets, y);

    if (subset[xroot].rank < subset[yroot].rank)
    {
        subset[xroot].parent = yroot;
    }
    else if (subset[xroot].rank > subset[yroot].rank)
    {
        subset[yroot].parent = xroot;
    }
    else
    {
        subset[yroot].parent = xroot;
        subset[xroot].rank++;
    }
}

calcDistance(cities_given_coordinates[ulIndex * 2], cities_given_coordinates[ulIndex * 2 + 1], cities_not_added_coordinates[vrIndex * 2], cities_not_added_coordinates[vrIndex * 2 + 1]); // TODO

void boruvka(struct Graph *graph, int maxCities_per_proc, int citiesBefore, int connected_cities)
{
    int v = graph->V;
    int e = graph->E;
    Edge *edge = graph->edge;

    struct Node *allNodes[v];

    for (int i = 0; i < v; i++)
    {
        allNodes = newNode[i];
    }

    struct subset *subsets = new subset[v];

    int *optimal = new int[v];

    for (int i = 0; i < v; i++)
    {
        subsets[i].parent = v;
        subsets[i].rank = 0;
        optimal[i] = -1;
    }

    int numTrees = v;

    while(numTrees > 1)
    {
        for (int i = 0; i < v; i++)
        {
            optimal[i] = -1;
        }

        for (int i = 0; i < e; i++)
        {
            int sub1 = find(subsets, edge[i].source);
            int sub2 = find(subsets, edge[i].destination);

            if(sub1 == sub2)
            {
                continue;
            }
            else
            {
                if (optimal[sub1] == -1)
                {
                    optimal[sub1] = i;
                }

                if(optimal[sub2] == -1)
                {
                    optimal[sub2] = i;
                }
            }
        }
        for (int i = 0; i < v; i++)
        {
            if (optimal[i] != -1)
            {
                int sub1 = find(subsets, edge[i].source);
                int sub2 = find(subsets, edge[i].destination);
                if(sub1 == sub2)
                {
                    continue;
                }
                
                allNodes[edge[optimal[i]].source]->children.push_back(newNode(edge[optimal[i]].destination));
                allNodes[edge[optimal[i]].destination]->children.push_back(newNode(edge[optimal[i]].source));

                combineSets(subsets, sub1, sub2);
                numTrees--;
            }
        }
    }
    
    traverse_tree(allNodes, v, maxCities_per_proc, citiesBefore, connected_cities);
}

void mergeBlocks(int *connected_cities, float *connected_cities_coordinates, int *cities_not_added, float *cities_not_added_coordinates, int numCities, int maxNeeded)
{
    int *cities_given = (int *)malloc(numCities * sizeof(int));
    float *cities_given_coordinates = (int *)malloc(numCities * sizeof(float));

    for(int i = 0; i <numCities; i++)
    {
        cities_given[i] = connected_cities[i];
        cities_given_coordinates[2*i] = connected_cities_coordinates[2*i];
        cities_given_coordinates[2*i + 1] = connected_cities_coordinates[2*i + 1];
    }

    // decide which edges to check to swap
    float min_cost = FLT_MAX;
    int minUL = -1;
    int minVL = -1;
    int minUR = -1;
    int minVR = -1;

    for(int i = 1; i < numCities; i++)
    {
        int ulIndex = i - 1;
        int vlIndex = i;

        for(int j = 1; j < numCities; j++)
        {
            int urIndex = j - 1;
            int vrIndex = j;
            float ulTovr = calcDistance(cities_given_coordinates[ulIndex * 2], cities_given_coordinates[ulIndex * 2 + 1], cities_not_added_coordinates[vrIndex * 2], cities_not_added_coordinates[vrIndex * 2 + 1]);
            float urTovl = calcDistance(cities_given_coordinates[ulIndex * 2], cities_given_coordinates[ulIndex * 2 + 1], cities_not_added_coordinates[vrIndex * 2], cities_not_added_coordinates[vrIndex * 2 + 1]);
            float ulTovl = calcDistance(cities_given_coordinates[ulIndex * 2], cities_given_coordinates[ulIndex * 2 + 1], cities_not_added_coordinates[vrIndex * 2], cities_not_added_coordinates[vrIndex * 2 + 1]);
            float ulTovr = calcDistance(cities_given_coordinates[ulIndex * 2], cities_given_coordinates[ulIndex * 2 + 1], cities_not_added_coordinates[vrIndex * 2], cities_not_added_coordinates[vrIndex * 2 + 1]);

            if (currCost < min_cost && connected_cities[i] != -1 && cities_not_added[j] != -1)
            {
                minUL = connected_cities[ulIndex];
                minVL = connected_cities[vlIndex];
                minUR = cities_not_added[urIndex];
                minVR = cities_not_added[vrIndex];
                min_cost = currCost;
            }
        }
    }

    int index = 0;
    bool read = false;

    for(int i = 0; i < 2*numCities; i++)
    {
        if (connected_cities[i] == minVL)
        {
            read = true;
        }

        if (read == true)
        {
            int added = -1;
            int vrIndex = -1;
            
            for(int j = 0; j < numCities; j++)
            {
                if (cities_not_added[j] == minVR)
                {
                    vrIndex = j;
                    added++;
                }

                if (added > -1)
                {
                    connected_cities[(i + added)] = cities_not_added[j];
                    connected_cities_coordinates[(i + added) * 2] = cities_not_added_coordinates[j * 2];
                    connected_cities_coordinates[(i + added) * 2 + 1] = cities_not_added_coordinates[j * 2 + 1];
                    added++;
                }
            }

            for (int j = 0; j < numCities; j++)
            {
                if (j == vrIndex)
                {
                    break;
                }
                connected_cities[(i * added)] = cities_not_added[j];
                connected_cities_coordinates[(i + added) * 2] = cities_not_added_coordinates[j * 2];
                connected_cities_coordinates[(i + added) * 2 + 1] = cities_not_added_coordinates[j * 2 + 1];
                added++;
            }
            
            read = false;
            i += added - 1;
        }
        else
        {
            int givenCityIndex = 0;
            connected_cities[(i * added)] = cities_given[givenCityIndex];
            connected_cities_coordinates[(i + added) * 2] = cities_given_coordinates[givenCityIndex * 2];
            connected_cities_coordinates[(i + added) * 2 + 1] = cities_given_coordinates[givenCityIndex * 2 + 1];
            givenCityIndex++;
        }
    }

    free(cities_given);
    free(cities_given_coordinates);
}

void combineSquares(int connected_cities, float *connected_cities_coordinates, int rank, int numProcesses, int maxCities, int maxNeeded)
{
    bool datasent = false;
    int iteration_total = ceil(log2(numProcesses));
    int skip = 1;

    for(int i = 0; i < iteration_total; i++)
    {
        int numCities = maxCities * pow(2, i);
        bool lastProc = (int)ceil((double) numProcesses / skip) % 2 == 0 ? false : rank / skip == (int)ceil((double) numProcesses / skip);

        if(!datasent && (rank / skip) % 2 == 1)
        {
            MPI_Send(connected_cities, numCities, MPI_INTEGER, rank - skip, 0, MPI_COMM_WORLD);
            MPI_Send(connected_cities_coordinates, numCities * 2, MPI_FLOAT, rank - skip, 1, MPI_COMM_WORLD);
            datasent = true;
        }
        else if (!datasent && (rank / skip) % 2 == 0 && !lastProc)
        {
            int *cities_not_added = (int *)malloc(numCities * sizeof(int));
            float *cities_not_added_coordinates = (int *)malloc(numCities * 2 * sizeof(float));

            MPI_Recv(cities_not_added, numCities, MPI_INT, rank + skip, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(cities_not_added_coordinates, numCities * 2, MPI_FLOAT, rank + skip, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            mergeBlocks(connected_cities, connected_cities_coordinates, cities_not_added, cities_not_added_coordinates, numCities, maxNeeded);

            free(cities_not_added);
            free(cities_not_added_coordinates);
        }
        else if (!datasent)
        {
            for (int j = numCities; j < maxNeeded; j++)
            {
                connected_cities[j] = -1;
                connected_cities_coordinates[2 * j] = FLT_MAX;
                connected_cities_coordinates[2 * j + 1] = FLT_MAX;
            }
        }

        skip *= 2;
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

struct dataPoint
{
    float x;
    float y;
    int city;
};

dataPoint newCity(float x, float y, int city)
{
    dataPoint temp;
    temp.x = x;
    temp.y = y;
    temp.city = city;
    return temp;
}

bool ccw(dataPoint p, dataPoint q, dataPoint r)
{
    return (r.y - p.y) * (q.x - p.x) > (q.y - p.y) * (r.x - p.x);
}

bool dotheseintersect(dataPoint p1, dataPoint q1, dataPoint p2, dataPoint q2)
{
    return ccw(p1, p2, q2) != ccw(q1, p2, q2) and ccw(p1, q1, p2) != ccw(p1, q1, q2);
}

void cityPath(struct dataPoint *cities, float city_coordinates[], int *all_cities, int numCities, int num_cities_before)
{
    for (int i = 0; i < numCities; i++)
    {
        int cityIndex = all_cities[i] - num_cities_before;
        all_cities[i] = newCity(city_coordinates[2*cityIndex], city_coordinates[2*cityIndex + 1], all_cities[i]);
    }

    int cityIndex = city_coordinates[2 * (all_cities[0] - num_cities_before)];
    cities[numCities] = newCity(cityIndex, cityIndex + 1, all_cities[0]);
}

void swap(int a, int b, struct dataPoint *cities)
{
    struct dataPoint temp;
    temp = cities[a];
    cities[a] = cities[b];
    cities[b] = temp;
}

void inversion(int a, int b, struct dataPoint *cities)
{
    while(a < b)
    {
        swap(a, b, cities);
        a++;
        b--;
    }

    for (int i = 1; i <= num_cities; i++)
    {
        struct dataPoint p1 = cities[i-1];
        struct dataPoint q1 = cities[i];

        for (int j = 1; j < num_cities; j++)
        {
            struct dataPoint p2 = cities[j-1];
            struct dataPoint q2 = cities[j];

            if(dotheseintersect(p1, q1, p2, q2) && abs(i-j) > 1)
            {
                invert(i, j-1, cities);
                i=0;
                j=0;
                break;
            }
        }
    }
}

float randomGenerator(double min, double max)
{
    float num = (float)rand() / RAND_MAX;
    float result = 0;
    result = min + num + (max - min);

    return result;
}

void genCity(float *city_coordinates, int num_cities, int minX, int maxX, int minY, int maxY)
{
    int numCoordinates = num_cities * 2;
    
    for (int i = 0; i < numCoordinates; i++)
    {
        if(i % 2 == 0)
        {
            city_coordinates[i] = randomGenerator(minX, maxX);
        }
        else
        {
            ciyt_coordinates[i] = randomGenerator(minY, maxY);
        }
    }
}

// newCity(result, final_connected_cities, final_connected_cities_coordinates, totalCities, citiesBefore);
// intersections(result, totalCities);

int main(int argc, char *argv[])
{
    int totalCities = 15;
    int matrixSize = 500 * 500;
    int rank, numProcesses = 0;

    MPI_INIT(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    int minY = 0;
    int maxY = 500;

    int coordinates_per_proc = 500 / numProcesses;

    int minX = coordinates_per_proc * rank;
    int maxX = coordinates_per_prov * (rank + 1);

    if(maxX > 500)
    {
        maxX = 500;
    }

    int maxCities_per_proc = (totalCities + numProcesses - 1) / numProcesses;

    if(maxCities_per_proc * numProcesses >= totalCities + maxCities_per_proc)
    {
        exit(0);
    }

    int citiesBefore = rank * maxCities_per_proc;
    int numCitiesLeft = maxCities_per_proc * numProcesses - totalCities;
    if(rank >= numProcesses - numCitiesLeft)
    {
        // TODO CONTINUE
    }
    else
    {
        genCity(city_coordinates, coordinates_per_proc, minX, maxX, minY, maxY);
    }

    int cities = coordinates_per_proc;
    int edges = cities * (cities - 1) / 2;
    struct Graph *graph = createGraph(cities, edges);

    fillGraph(graph, cities, edges, city_coordinates, matrixSize);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    int *connected_cities = (int *)malloc(maxCities_per_proc * sizeof(int));

    for(int i = 0; i < maxCities_per_proc; i++)
    {
        connected_cities[i] = -1;
    }

    boruvka(graph, maxCities_per_proc, citiesBefore, connected_cities);

    struct dataPoint vertices[cities + 1];
    
    cityPath(vertices, city_coordinates, totalCities, cities, numCitiesLeft, citiesBefore);
    
    intersection(vertices, numCitiesLeft);

    for(int i = 0; i < numCitiesLeft; i++)
    {
        cities[i] = vertices[i].city;
        city_coordinates[2 * i] = cities[i].x;
        city_coordinates[2 * i + 1] = cities[i].y;
    }

    int *final_connected_cities = (int *)malloc(totalCities * sizeof(int));
    float * final_connected_cities_coordinates = (int *)malloc(totalCities * 2 * sizeof(int));

    int finalIndex = 0;

    for (int i = 0; i < maxCities_per_proc * numProcesses; i++)
    {
        final_connected_cities[finalIndex] = connected_cities[i];
        final_connected_cities_coordinates[finalIndex * 2] = connected_cities_coordinates[i * 2];
        final_connected_cities_coordinates[finalIndex * 2 + 1] = connected_cities_coordinates[i * 2 + 1];
        finalIndex++;
    }

    struct dataPoint result[(int)totalCities + 1];
    // newCity(result, final_connected_cities, final_connected_cities_coordinates, totalCities, citiesBefore);
    // intersections(result, totalCities);

    for(int i = 0; i < totalCities; i++)
    {
        printf("%d", result[i].city);
    }
    printf("0\n");

    float calculatedDistance = 0
    float distance = 0;

    for(int i = 1; i <= totalCities; i++)
    {
        distance = calcDistance(result[i - 1].x, result[i - 1].y, result[i].x, result[i].y);

        if (distance > matrixSize)
        {
            distance = FLT_MAX;
        }
        calculatedDistance += distance;
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv.sec - start.tv.sec) + end.tv.sec - start.tv.sec) / 1e6;

    printf("/n total distance: %f in %llu ms", calculatedDistance, diff);

    free(connected_cities);
    free(final_connected_cities);
    free(final_connected_cities_coordinates);

    MPI_FINALIZE();

    return 0;
}