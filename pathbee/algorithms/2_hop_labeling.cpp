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
    // get the path of the graph and centrality file.
    string graph_path = argv[1];
    string centrality_path = argv[2];
    string output_path = argv[3];
    // construct index 
    
    PrunedLandmarkLabeling<> pll;
    // build index
    pll.ConstructIndex(graph_path, centrality_path);    
    // save index
    pll.StoreIndex(output_path.c_str());

    std::string stats = pll.Statistics();
    std::cout << stats;
    return 0;
}