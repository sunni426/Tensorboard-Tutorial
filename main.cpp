//
//  main.cpp
//  graph
//
//  Copyright Â© 2022 Tali Moreshet. All rights reserved.
//

#include "Graph.h"

int main(int argc, const char * argv[]) {
    
    if (argc != 2)
    {
        cout << "Please supply a file name as input" << endl;
        return -1;
    }
    
    Graph graph;
    graph.generateGraph(argv[1]);
    graph.printGraph();
    graph.lowestReachable();
    
    return 0;
}