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
    cout << "Enter queries in format 'start end' (or -1 to exit):" << endl;

    int v, w;
    while (true) {
        cin >> v;
        if (v == -1) break;
        cin >> w;
        if (w == -1) break;

        int distance = pll.QueryDistance(v, w);
        if (distance == INT_MAX) {
            cout << "No path exists between vertices " << v << " and " << w << endl;
        } else {
            cout << "Distance between vertices " << v << " and " << w << " is: " << distance << endl;
        }
    }

    return 0;
} 