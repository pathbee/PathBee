#include "pruned_indexing.h"
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>

using namespace std;

void construct_index(const string& graph_path, const string& centrality_path, const string& output_path) {
    PrunedLandmarkLabeling<> pll;
    pll.ConstructIndex(graph_path, centrality_path);
    pll.StoreIndex(output_path.c_str());
    std::string stats = pll.Statistics();
    std::cout << stats << std::endl;
}

void query_distance(const string& index_path, const string& query_file, const string& output_file) {
    PrunedLandmarkLabeling<> pll;
    if (!pll.LoadIndex(index_path.c_str())) {
        cout << "Failed to load index from " << index_path << endl;
        return;
    }
    ifstream fin(query_file);
    ofstream fout(output_file);
    int v, w;
    while (fin >> v >> w) {
        auto start = std::chrono::high_resolution_clock::now();
        int distance = pll.QueryDistance(v, w);
        auto end = std::chrono::high_resolution_clock::now();
        double query_time = std::chrono::duration<double, std::micro>(end - start).count();
        fout << v << ' ' << w << ' ' << distance << ' ' << query_time << '\n';
    }
}

void query_index_items(const string& index_path) {
    PrunedLandmarkLabeling<> pll;
    if (!pll.LoadIndex(index_path.c_str())) {
        cout << "Failed to load index from " << index_path << endl;
        return;
    }
    cout << "Index loaded successfully. Number of vertices: " << pll.GetNumVertices() << endl;
    cout << "Enter vertex id to query its index item count (or -1 to exit):" << endl;
    int v;
    while (true) {
        cin >> v;
        if (v == -1) break;
        auto counts = pll.GetNumIndexItems(v);
        cout << "Vertex " << v << ": inIndex items = " << counts.first
             << ", outIndex items = " << counts.second << endl;
    }
}

void batch_query_distance(const string& index_path, const string& query_file, const string& output_file) {
    PrunedLandmarkLabeling<> pll;
    if (!pll.LoadIndex(index_path.c_str())) {
        cout << "Failed to load index from " << index_path << endl;
        return;
    }
    ifstream fin(query_file);
    ofstream fout(output_file);
    int v, w;
    while (fin >> v >> w) {
        auto start = std::chrono::high_resolution_clock::now();
        int distance = pll.QueryDistance(v, w);
        auto end = std::chrono::high_resolution_clock::now();
        double query_time = std::chrono::duration<double, std::micro>(end - start).count();
        fout << v << ' ' << w << ' ' << distance << ' ' << query_time << '\n';
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage:\n"
             << "  " << argv[0] << " construct <graph_path> <centrality_path> <output_path>\n"
             << "  " << argv[0] << " distance <index_file> <query_file> <output_file>\n"
             << "  " << argv[0] << " index_items <index_file>\n"
             << "  " << argv[0] << " batch_distance <index_file> <query_file> <output_file>\n";
        return 1;
    }

    string mode = argv[1];

    if (mode == "construct") {
        if (argc != 5) {
            cout << "Usage: " << argv[0] << " construct <graph_path> <centrality_path> <output_path>\n";
            return 1;
        }
        construct_index(argv[2], argv[3], argv[4]);
    } else if (mode == "distance") {
        if (argc != 5) {
            cout << "Usage: " << argv[0] << " distance <index_file> <query_file> <output_file>\n";
            return 1;
        }
        query_distance(argv[2], argv[3], argv[4]);
    } else if (mode == "index_items") {
        if (argc != 3) {
            cout << "Usage: " << argv[0] << " index_items <index_file>\n";
            return 1;
        }
        query_index_items(argv[2]);
    } else if (mode == "batch_distance") {
        if (argc != 5) {
            cout << "Usage: " << argv[0] << " batch_distance <index_file> <query_file> <output_file>\n";
            return 1;
        }
        batch_query_distance(argv[2], argv[3], argv[4]);
    } else {
        cout << "Unknown mode: " << mode << endl;
        return 1;
    }
    return 0;
}