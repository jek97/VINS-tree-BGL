#ifndef TREE_H
#define TREE_H

#include <string>
#include <map>
#include <vector>
#include <opencv2/core.hpp>

class TreeNode{
    public:
        int id = -1; // internal id
        double x; // position x
        double y; // position y
        double z; // position z
        double v_x = 0; // velocity x 
        double v_y = 0; // velocity y 
        double v_z = 0; // velocity z 
        double n_x = 0; // vector_to_parent node (z versor for root)
        double n_y = 0; // vector_to_parent node (z versor for root)
        double n_z = 0; // vector_to_parent node (z versor for root)
        int track_cnt; // tracking counter (how many consecutive frames we've seen the feature)
};

class Ex_TreeNode{
    public:
        std::string ex_id; //external id assigned to the node
        int id = -1; // internal id
        double x; // position x
        double y; // position y
        double z; // position z
        double v_x = 0; // velocity x 
        double v_y = 0; // velocity y 
        double v_z = 0; // velocity z 
        std::vector<double> fd; // feature descriptor
        std::vector<uint8_t> fd_brief;
        std::string ex_parent; // extenal parent node id
        std::vector<std::string> ex_sons; // external sons node ids
        std::vector<std::string> extended_sons; // extended sons of a node, coded as the nodes from the current node to the tip
        int component = -1; // variable to store to which component is the node belonging to
        int track_cnt; // tracking counter (how many consecutive frames we've seen the feature)
};

class Ex_TreeNode_Skel{
    public:
        cv::Point ex_id; //external id assigned to the node
        cv::Point ex_parent; // extenal parent node id
        std::vector<cv::Point> ex_sons; // external sons node ids
        int component = -1; // variable to store to which component is the node belonging to
};


#endif