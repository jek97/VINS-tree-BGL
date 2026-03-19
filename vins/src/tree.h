#ifndef TREE_H
#define TREE_H

#include <string>
#include <map>
#include <vector>
#include <opencv2/core.hpp>
#include <boost/graph/adjacency_list.hpp>

class Ex_TreeNode_Skel{
    public:
        cv::Point ex_id; //external id assigned to the node
        cv::Point ex_parent; // extenal parent node id
        std::vector<cv::Point> ex_sons; // external sons node ids
        int component = -1; // variable to store to which component is the node belonging to
};

// Vertex bundle for the tracker-side graph.
struct ObservedNode
{
    std::string ex_id;         // external id assigned to the node
    int         id      = -1;  // internal id
    double x   = 0.0;
    double y   = 0.0;
    double z   = 0.0;
    double v_x = 0.0;
    double v_y = 0.0;
    double v_z = 0.0;
    std::vector<double>   fd;
    std::vector<uint8_t>  fd_brief;
    int component = -1;
    int track_cnt =  0;
};

using ObservedTree   = boost::adjacency_list<
    boost::vecS, boost::vecS, boost::bidirectionalS,
    ObservedNode, boost::no_property
>;
using ObservedForest = std::vector<ObservedTree>;


#endif
