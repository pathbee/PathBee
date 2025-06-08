#include "pruned_indexing.h"
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>

using namespace std;

class IndexManager {
private:
    PrunedLandmarkLabeling<> pll;
    bool is_loaded = false;
    string current_index_path;

public:
    bool loadIndex(const string& index_path) {
        if (is_loaded && current_index_path == index_path) {
            return true;
        }
        if (!pll.LoadIndex(index_path.c_str())) {
            cout << "Failed to load index from " << index_path << endl;
            return false;
        }
        is_loaded = true;
        current_index_path = index_path;
        return true;
    }

    pair<int, int> getIndexItems(int vertex_id) {
        if (!is_loaded) {
            cout << "Debug - Index not loaded!" << endl;
            return make_pair(-1, -1);
        }
        if (vertex_id < 0 || vertex_id >= pll.GetNumVertices()) {
            cout << "Debug - Invalid vertex ID: " << vertex_id << endl;
            return make_pair(-1, -1);
        }
        return pll.GetNumIndexItems(vertex_id);
    }

    int queryDistance(int v, int w) {
        if (!is_loaded) {
            return -1;
        }
        return pll.QueryDistance(v, w);
    }

    int getNumVertices() {
        return pll.GetNumVertices();
    }
};

// Global index manager
static IndexManager index_manager;

void construct_index(const string& graph_path, const string& centrality_path, const string& output_path) {
    PrunedLandmarkLabeling<> pll;
    pll.ConstructIndex(graph_path, centrality_path);
    pll.StoreIndex(output_path.c_str());
    std::string stats = pll.Statistics();
    std::cout << stats << std::endl;
}

void query_distance(const string& index_path, const string& query_file, const string& output_file) {
    if (!index_manager.loadIndex(index_path)) {
        return;
    }
    ifstream fin(query_file);
    ofstream fout(output_file);
    int v, w;
    while (fin >> v >> w) {
        auto start = std::chrono::high_resolution_clock::now();
        int distance = index_manager.queryDistance(v, w);
        auto end = std::chrono::high_resolution_clock::now();
        double query_time = std::chrono::duration<double, std::micro>(end - start).count();
        fout << v << ' ' << w << ' ' << distance << ' ' << query_time << '\n';
    }
}

void query_index_items(const char* index_path, const char* vertex_file) {
    if (!index_manager.loadIndex(index_path)) {
        cerr << "Failed to load index from " << index_path << endl;
        return;
    }

    // Read vertices from file
    ifstream fin(vertex_file);
    if (!fin.is_open()) {
        cerr << "Failed to open vertex file: " << vertex_file << endl;
        return;
    }

    int vertex_id;
    while (fin >> vertex_id) {
        if (vertex_id >= index_manager.getNumVertices()) {
            cerr << "Invalid vertex ID: " << vertex_id << endl;
            continue;
        }
        auto counts = index_manager.getIndexItems(vertex_id);
        cout << "Vertex " << vertex_id << ": in=" << counts.first 
             << ", out=" << counts.second << endl;
    }
}

void batch_query_distance(const string& index_path, const string& query_file, const string& output_file) {
    if (!index_manager.loadIndex(index_path)) {
        return;
    }
    ifstream fin(query_file);
    ofstream fout(output_file);
    int v, w;
    while (fin >> v >> w) {
        auto start = std::chrono::high_resolution_clock::now();
        int distance = index_manager.queryDistance(v, w);
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
             << "  " << argv[0] << " index_items <index_file> <vertex_file>\n"
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
        if (argc != 4) {
            cout << "Usage: " << argv[0] << " index_items <index_file> <vertex_file>\n";
            return 1;
        }
        query_index_items(argv[2], argv[3]);
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