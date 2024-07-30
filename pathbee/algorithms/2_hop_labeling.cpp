#include "pruned_indexing.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <filesystem>
#include <random>
#include <chrono>

using namespace std;
char* choose_centrality(int n);

int main(int argc, char** argv){
    // get the mapPath and centralityPath

    string map_path = argv[1];
    string centrality_path = argv[2];
    
    // construct index 
    
    PrunedLandmarkLabeling<> pll;
    // build index
    pll.ConstructIndex(map_path, centrality_path);

    
    std::string stats = pll.Statistics();
    // query 100W times
    int V = pll.GetNumVertices();
    mt19937_64 rng(random_device{}());
    uniform_int_distribution<int> dist(0, V-1);
    vector<int> rand_nums;
    auto start_time = chrono::high_resolution_clock::now();
    for (int i=0; i< 1000000; i++){
        int start = dist(rng);
        int end = dist(rng);
        pll.QueryDistance(dist(rng), dist(rng));
    }
    auto end_time = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
    stats += " 100W times query time:" + std::to_string(elapsed.count()) + "us\n";
    std::cout << stats;
    
    return 0;
}