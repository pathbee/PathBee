#include "pruned_indexing.h"
#include <iostream>
#include <string>

using namespace std;

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <index_file>" << endl;
        return 1;
    }

    string index_path = argv[1];

    // Create PLL object and load the index
    PrunedLandmarkLabeling<> pll;
    if (!pll.LoadIndex(index_path.c_str())) {
        cout << "Failed to load index from " << index_path << endl;
        return 1;
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

    return 0;
}
