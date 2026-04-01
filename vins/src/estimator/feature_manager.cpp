/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}


FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
    t_feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {
        it.used_num = it.feature_per_frame.size();
        if (it.used_num >= 4)
        {
            cnt++;
        }
    }
    // no need to count the tree features

    return cnt;
}

int FeatureManager::get_tree_FeatureCount()
{
    int cnt = 0;
    for (auto &tree : t_feature)
        for (int v = 0; v < (int)boost::num_vertices(tree); ++v)
        {
            ModelNode &node = tree[v];
            node.used_num = node.tree_per_frame.size();
            if (node.used_num >= 3)
                cnt++;
        }
    return cnt;
}

void FeatureManager::logMessage(const std::string& message) 
{
    const std::string LOG_FILE_PATH = "/home/glugano/Desktop/log.txt";

    std::ofstream logFile(LOG_FILE_PATH, std::ios::app);
    if (!logFile) {
        std::cerr << "Error: Unable to open log file." << std::endl;
        return;
    }

    logFile << message << std::endl;
}

bool FeatureManager::addFeatureTreeCheckParallax(int frame_count, const double header, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const pair<double, vector<pair<int, ObservedTree>>> &tree, double td, const Eigen::Matrix3d* Rs, const Eigen::Vector3d* Ps, const Eigen::Matrix3d& ric0, const Eigen::Vector3d& tic0)
{
    ///// LOG /////
    std::ostringstream oss;
    oss << "=========================================================================\nFM adding features to buffer at frame count " << frame_count << "\nnormal features: " << std::endl;

    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0; // number of feature occured in this frame and last frame
    double parallax_tree_sum = 0;
    int parallax_tree_num = 0; // number of feature occured in this frame and last frame
    last_track_num = 0; // number of features that has been already found in the previous frames
    last_average_parallax = 0;
    new_feature_num = 0; // number of new feature encountered  with the current frame
    long_track_num = 0; // number of features that has been already encountered in the previous frames, for at least 4 frames
    for (auto &id_pts : image) // for every point in the image
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td); // add the point (with x, y, z, p_u, p_v, v_x, v_y) in the f_per_frame variable together with the time offset td
        assert(id_pts.second[0].first == 0); // meaning the camera id is 0 (first camera)
        if(id_pts.second.size() == 2) // meaning that there are two camera ids and so stereocamera case
        {
            f_per_fra.rightObservation(id_pts.second[1].second); // add the points (obtained from the second camera) in the f_per_frame variable with the time offset td
            assert(id_pts.second[1].first == 1); // meaning the camera id is 1 (second camera)
        }

        int feature_id = id_pts.first; // get the feature_id (same as the one received from the message)
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it) 
                          {
            return it.feature_id == feature_id;
                          }); // search in the feature vector, form the beginning to the end if the feature id is present, it return a pointer to the element in the feature vector that has the same feature_id, or if no such element exists it return feature.end()

        if (it == feature.end()) // if it's the first time we encounter a feature with that feature_id
        {   
            oss << "    new feature " << feature_id << " add feature " << std::endl;
            feature.push_back(FeaturePerId(feature_id, frame_count)); // add the feature_id in the feature vector, with the associated frame_count (counting the first frame in which we saw this feature)
            feature.back().feature_per_frame.push_back(f_per_fra); // in this feature, under the feature_per_frame field add the point data (x, y, z, p_u, p_v, v_x, v_y)
            new_feature_num++; // increase the number of new feature encountered  with the current frame
        }
        else if (it->feature_id == feature_id) // if the feature was already in the feature vector
        {
            oss << "    old feature " << feature_id << " add observation" << std::endl;
            it->feature_per_frame.push_back(f_per_fra); // in this feature, under the feature_per_frame field add the point data (x, y, z, p_u, p_v, v_x, v_y) obtained in the current frame
            last_track_num++; // increase the number of features that has been already found in the previous frames
            if( it-> feature_per_frame.size() >= 4) // if the feature has been encounter in at least 4 previous frames
                long_track_num++; // increase the number of features that has been already encountered in the previous frames, for at least 4 frames
        }
    }

    // --- Tree features ---
    int last_t_track_num   = 0;
    int new_t_feature_num  = 0;
    int long_t_track_num   = 0;
    for (const auto &[prev_idx, obs_tree] : tree.second)
    {
        if (prev_idx == -1)
        {
            // Unmatched tree: add as a brand-new ModelTree to t_feature.
            ModelTree model_tree;

            // Add vertices: one ModelNode per ObservedNode, preserving all fields.
            for (int v = 0; v < (int)boost::num_vertices(obs_tree); ++v)
            {
                const ObservedNode &obs_node = obs_tree[v];

                ModelNode model_node(obs_node.id, frame_count);
                model_node.used_num        = 0;
                model_node.estimated_depth = -1.0;
                model_node.solve_flag      = 0;
                model_node.tree_per_frame.emplace_back(obs_node, td, frame_count);

                boost::add_vertex(model_node, model_tree);
                new_t_feature_num++;
            }

            // Mirror edges from the ObservedTree, preserving parent-son relations.
            auto [ei, ei_end] = boost::edges(obs_tree);
            for (; ei != ei_end; ++ei)
                boost::add_edge(boost::source(*ei, obs_tree),
                                boost::target(*ei, obs_tree), model_tree);

            t_feature.push_back(std::move(model_tree));
        }
        else
        {
            // Matched tree: analyse topological consistency between the current
            // observed tree and the existing ModelTree at t_feature[prev_idx].
            ModelTree &model_tree = t_feature[prev_idx];

            // --- helpers ---

            // All obs nodes that are tracked (track_cnt > 1): id -> vd in obs_tree.
            unordered_map<int, int> matched_obs; // id -> obs vd
            for (int v = 0; v < (int)boost::num_vertices(obs_tree); ++v)
                if (obs_tree[v].track_cnt > 1 && obs_tree[v].id >= 0)
                    matched_obs[obs_tree[v].id] = v;

            // Find vd in model_tree by feature_id.
            auto find_model_vd = [&](int id) -> int {
                for (int v = 0; v < (int)boost::num_vertices(model_tree); ++v)
                    if (model_tree[v].feature_id == id) return v;
                return -1;
            };

            // Walk toward root in obs_tree from start_vd.
            // Returns {path, cb_vd} where path = [start_vd ... cb_vd] and
            // cb_vd is the first ancestor with track_cnt > 1 (-1 if root reached first).
            auto walk_obs = [&](int start_vd) -> pair<vector<int>, int> {
                vector<int> path = {start_vd};
                int curr = start_vd;
                while (true) {
                    auto [ie, ie_end] = boost::in_edges(curr, obs_tree);
                    if (ie == ie_end) return {path, -1};          // reached root
                    curr = (int)boost::source(*ie, obs_tree);
                    path.push_back(curr);
                    if (obs_tree[curr].track_cnt > 1) return {path, curr}; // found CB
                }
            };

            // Walk toward root in model_tree from start_vd.
            // A model node is "matched" if its feature_id is in matched_obs.
            // Returns {path, mb_vd} where mb_vd = -1 if root reached first.
            auto walk_model = [&](int start_vd) -> pair<vector<int>, int> {
                vector<int> path = {start_vd};
                int curr = start_vd;
                while (true) {
                    auto [ie, ie_end] = boost::in_edges(curr, model_tree);
                    if (ie == ie_end) return {path, -1};          // reached root
                    curr = (int)boost::source(*ie, model_tree);
                    path.push_back(curr);
                    if (matched_obs.count(model_tree[curr].feature_id)) return {path, curr}; // found MB
                }
            };

            // --- topological consistency analysis ---

            // key   = CB node id
            // value = list of (path CB→CA in obs, path MB→MA in model)
            //         where each path is a sequence of vertex descriptors
            //         ordered from ancestor (CB/MB) down to descendant (CA/MA).
            unordered_map<int, vector<pair<vector<int>, vector<int>>>> path_map;

            // nodes CA (obs vd) whose parent topology is inconsistent with the model.
            vector<int> topological_violations;

            for (const auto &[ca_id, ca_vd] : matched_obs)
            {
                int ma_vd = find_model_vd(ca_id);
                if (ma_vd < 0) continue; // model node not found (shouldn't happen)

                auto [path_obs,   cb_vd] = walk_obs  (ca_vd);
                auto [path_model, mb_vd] = walk_model(ma_vd);

                // If either walk hit the root without finding a matched ancestor
                // there is nothing to compare: skip.
                if (cb_vd < 0 || mb_vd < 0) continue;

                int cb_id          = obs_tree[cb_vd].id;
                int mb_feature_id  = model_tree[mb_vd].feature_id;

                if (cb_id != mb_feature_id)
                {
                    // CB and MB are different nodes: topological violation.
                    topological_violations.push_back(ca_vd);
                }
                else
                {
                    // CB and MB are the same tracked node: consistent ancestry.
                    // Store path CB→CA (reverse of obs walk) and MB→MA (reverse of model walk).
                    vector<int> path_cb_to_ca(path_obs.rbegin(),   path_obs.rend());
                    vector<int> path_mb_to_ma(path_model.rbegin(), path_model.rend());
                    path_map[cb_id].emplace_back(std::move(path_cb_to_ca),
                                                 std::move(path_mb_to_ma));
                }
            }

            // obs_vd -> model_vd for every obs node that has or gets a model counterpart.
            // Initialised for matched nodes; extended by Parts 3 and 1.
            unordered_map<int, int> obs_to_model_vd;
            for (const auto &[id, obs_vd] : matched_obs) {
                int mvd = find_model_vd(id);
                if (mvd >= 0) obs_to_model_vd[obs_vd] = mvd;
            }

            // Project a model node's last observation to world frame for sorting.
            auto to_world = [&](const ModelNode &node) -> Eigen::Vector3d {
                if (node.tree_per_frame.empty()) return Eigen::Vector3d::Zero();
                const TreePerFrame &tpf = node.tree_per_frame.back();
                return Rs[tpf.frame] * (ric0 * tpf.point + tic0) + Ps[tpf.frame];
            };

            // ----------------------------------------------------------------
            // PART 2 — topological violations: add new observation, keep topology
            // ----------------------------------------------------------------
            for (int ca_vd : topological_violations) {
                int ma_vd = find_model_vd(obs_tree[ca_vd].id);
                if (ma_vd >= 0)
                    model_tree[ma_vd].tree_per_frame.emplace_back(obs_tree[ca_vd], td, frame_count);
            }

            // ----------------------------------------------------------------
            // PART 3 — path reordering between pairs of matched anchors
            // ----------------------------------------------------------------
            // obs_vd of nodes that were Part-3 intermediates (excluded from Part 1).
            unordered_set<int> path_obs_intermediates;

            for (auto &[cb_id, pairs] : path_map) {
                int mb_vd = find_model_vd(cb_id);
                if (mb_vd < 0) continue;

                for (auto &[p_obs, p_model] : pairs) {
                    // p_obs  = [cb_vd, ..., ca_vd]  (obs_tree vds)
                    // p_model = [mb_vd, ..., ma_vd]  (model_tree vds)
                    if (p_obs.size() < 2 || p_model.size() < 2) continue;

                    int cb_vd  = p_obs.front();
                    int ca_vd  = p_obs.back();
                    int ma_vd  = p_model.back();

                    // Anchor positions used for distance computation.
                    Eigen::Vector3d cb_pos(obs_tree[cb_vd].x, obs_tree[cb_vd].y, obs_tree[cb_vd].z);
                    Eigen::Vector3d mb_world = to_world(model_tree[mb_vd]);

                    struct IntNode {
                        bool is_model;
                        int  orig_vd;   // vd in its own tree (obs or model)
                        int  model_vd;  // resolved model-tree vd (set during insertion)
                        double dist;
                    };
                    vector<IntNode> intermediates;

                    // Collect obs intermediates (indices 1 .. size-2).
                    for (int k = 1; k < (int)p_obs.size() - 1; ++k) {
                        int vd = p_obs[k];
                        path_obs_intermediates.insert(vd);
                        Eigen::Vector3d pos(obs_tree[vd].x, obs_tree[vd].y, obs_tree[vd].z);
                        intermediates.push_back({false, vd, -1, (pos - cb_pos).norm()});
                    }

                    // Collect model intermediates (indices 1 .. size-2).
                    for (int k = 1; k < (int)p_model.size() - 1; ++k) {
                        int vd = p_model[k];
                        Eigen::Vector3d wpos = to_world(model_tree[vd]);
                        intermediates.push_back({true, vd, vd, (wpos - mb_world).norm()});
                    }

                    // Sort ascending by distance; model nodes first at ties.
                    stable_sort(intermediates.begin(), intermediates.end(),
                        [](const IntNode &a, const IntNode &b) {
                            if (std::abs(a.dist - b.dist) > 1e-9) return a.dist < b.dist;
                            return (int)a.is_model > (int)b.is_model;
                        });

                    // Remove old edges along the model path.
                    for (int k = 0; k < (int)p_model.size() - 1; ++k) {
                        auto [ed, found] = boost::edge(p_model[k], p_model[k + 1], model_tree);
                        if (found) boost::remove_edge(ed, model_tree);
                    }

                    // Add new model vertices for obs intermediates; record mapping.
                    for (auto &inter : intermediates) {
                        if (!inter.is_model) {
                            const ObservedNode &on = obs_tree[inter.orig_vd];
                            ModelNode mn(on.id, frame_count);
                            mn.tree_per_frame.emplace_back(on, td, frame_count);
                            int new_mvd = (int)boost::add_vertex(mn, model_tree);
                            obs_to_model_vd[inter.orig_vd] = new_mvd;
                            inter.model_vd = new_mvd;
                        }
                    }

                    // Wire new chain: mb_vd → sorted intermediates → ma_vd.
                    int prev_mvd = mb_vd;
                    for (const auto &inter : intermediates) {
                        boost::add_edge(prev_mvd, inter.model_vd, model_tree);
                        prev_mvd = inter.model_vd;
                    }
                    boost::add_edge(prev_mvd, ma_vd, model_tree);

                    // Append new observations to both endpoint model nodes.
                    model_tree[mb_vd].tree_per_frame.emplace_back(obs_tree[cb_vd], td, frame_count);
                    model_tree[ma_vd].tree_per_frame.emplace_back(obs_tree[ca_vd], td, frame_count);
                }
            }

            // ----------------------------------------------------------------
            // PART 1 — unmatched nodes that descend from at least one matched node
            // ----------------------------------------------------------------
            // Mark every obs node whose subtree-path to root passes through a matched node.
            unordered_set<int> has_matched_ancestor;
            {
                function<void(int, bool)> dfs = [&](int v, bool anc_matched) {
                    if (anc_matched) has_matched_ancestor.insert(v);
                    bool child_anc = anc_matched || (obs_tree[v].track_cnt > 1);
                    for (auto e : boost::make_iterator_range(boost::out_edges(v, obs_tree)))
                        dfs((int)boost::target(e, obs_tree), child_anc);
                };
                for (int v = 0; v < (int)boost::num_vertices(obs_tree); ++v)
                    if (boost::in_degree(v, obs_tree) == 0)
                        dfs(v, false);
            }

            // BFS top-down so parents are always resolved before their children.
            {
                queue<int> q;
                for (int v = 0; v < (int)boost::num_vertices(obs_tree); ++v)
                    if (boost::in_degree(v, obs_tree) == 0) q.push(v);

                while (!q.empty()) {
                    int v = q.front(); q.pop();
                    for (auto e : boost::make_iterator_range(boost::out_edges(v, obs_tree)))
                        q.push((int)boost::target(e, obs_tree));

                    if (obs_tree[v].track_cnt > 1)         continue; // matched
                    if (path_obs_intermediates.count(v))   continue; // Part 3 intermediate
                    if (!has_matched_ancestor.count(v))    continue; // above all matches

                    const ObservedNode &on = obs_tree[v];
                    ModelNode mn(on.id, frame_count);
                    mn.tree_per_frame.emplace_back(on, td, frame_count);
                    int new_mvd = (int)boost::add_vertex(mn, model_tree);
                    obs_to_model_vd[v] = new_mvd;

                    // Connect to parent using the resolved model vd.
                    auto [ie, ie_end] = boost::in_edges(v, obs_tree);
                    if (ie != ie_end) {
                        int par_obs_vd = (int)boost::source(*ie, obs_tree);
                        auto it = obs_to_model_vd.find(par_obs_vd);
                        if (it != obs_to_model_vd.end())
                            boost::add_edge(it->second, new_mvd, model_tree);
                    }
                }
            }

            last_t_track_num++;
        }
    }

    ///// LOG /////
    std::ostringstream oss1;
    oss1 << "=========================================================================\nFM buffer update at frame_count=" << frame_count
         << "\nnormal: new=" << new_feature_num << " tracked=" << last_track_num << " long=" << long_track_num
         << "\ntree:   new_nodes=" << new_t_feature_num << " matched_trees=" << last_t_track_num
         << std::endl;
    oss << "tree: new_nodes=" << new_t_feature_num << std::endl;
    logMessage(oss.str());
    logMessage(oss1.str());
    ///// LOG /////

    std::ostringstream oss2;
    oss2 << "=========================================================================\nFM accumulator timer at frame count " << frame_count << std::endl;

    for (auto &it_per_id : feature) // for all the feature already saved, together with the newly discovered with this frame
    {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1) // if the feature first appearence is farthest then 2 frames from the current one and [the sum of the first appearence of the feature (in frames) plus the number of time we ennocuntered it is bigger then the actual frame count] = if the feature appeared the first time at least in the previous frame and it also appeared in the current frame
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count); // get the distance of the feature in the image between the last two frames in whic it appeared
            parallax_num++; // increase the number of feature occured in this frame and last frame
        }
    }
    // parallax contribution from tree nodes that have observations at frame_count-2 and frame_count-1
    for (const auto &model_tree : t_feature)
        for (int v = 0; v < (int)boost::num_vertices(model_tree); ++v)
        {
            const ModelNode &node = model_tree[v];
            if (node.has_frame(frame_count - 2) && node.has_frame(frame_count - 1))
            {
                parallax_tree_sum += compensated_tree_Parallax2(node, frame_count);
                parallax_tree_num++;
            }
        }

    bool fast_flag = false;
    bool var_flag = false;
    // low pass signals for debugging
    double alpha_1 = 0.9;
    filtered_last_track_num   = alpha_1 * last_track_num   + (1 - alpha_1) * filtered_last_track_num;
    filtered_last_t_track_num = alpha_1 * last_t_track_num + (1 - alpha_1) * filtered_last_t_track_num;
    filtered_long_track_num   = alpha_1 * long_track_num   + (1 - alpha_1) * filtered_long_track_num;
    filtered_long_t_track_num = alpha_1 * long_t_track_num + (1 - alpha_1) * filtered_long_t_track_num;
    filtered_new_feature_num  = alpha_1 * new_feature_num  + (1 - alpha_1) * filtered_new_feature_num;
    filtered_new_t_feature_num = alpha_1 * new_t_feature_num + (1 - alpha_1) * filtered_new_t_feature_num;

    // raw observed value
    double x_k = long_track_num + long_t_track_num;

    oss2 << "x_k " << x_k << " \nX effective max 0 " << X_effective_max << std::endl;

    // update effective max
    if (x_k > X_effective_max)
        X_effective_max = x_k;
    else
        X_effective_max -= gamma;

    X_effective_max = std::min(X_effective_max, static_cast<double>(MAX_CNT + MAX_T_CNT));

    double err = x_k - last_track_num_plateau;

    var_threshold = vt_K * std::sqrt(noise_var_estimate);
    oss2 << "X effective max " << X_effective_max << "\nerr " << err << " \nvar_threshold " << var_threshold << " \nvar_threshold_max " << var_threshold_max << std::endl;
    if (var_threshold > var_threshold_max)
    {
        var_threshold_max = var_threshold;
        oss2 << "    updated var_threshold_max to " << var_threshold_max << std::endl;
    }

    if (std::abs(err) > var_threshold)
    {
        fast_flag = true;
        oss2 << "    fast updated last_track_num_plateau from " << last_track_num_plateau;
        last_track_num_plateau += alpha_fast * err;
        oss2 << " to " << last_track_num_plateau << std::endl;
    }
    else
    {
        oss2 << "    slow updated last_track_num_plateau from " << last_track_num_plateau;
        last_track_num_plateau += alpha_slow * err;
        oss2 << " to " << last_track_num_plateau << std::endl;
        var_flag = true;
        oss2 << "    slow updated noise_var_estimate from " << noise_var_estimate;
        noise_var_estimate = (1 - betha) * noise_var_estimate + betha * (err * err);
        oss2 << " to " << noise_var_estimate << std::endl;
    }

    double D_max = std::max(0.0, (X_effective_max - last_track_num_plateau));
    double c_val = (accumulator_timer_thresh / delta_time_0) * (header - prev_time);
    prev_time = header;
    double rd_k   = ((accumulator_timer_thresh - c_val) / std::max(1.0, (X_effective_max - min_track_num - var_threshold_max))) + 1.2;
    double r_D_max = rd_k * std::max(0.0, D_max - var_threshold);

    accumulator_timer = std::max(0.0, accumulator_timer + r_D_max + c_val);
    oss2 << "D_max " << D_max << "\nrd_k " << rd_k << "\nr_D_max " << r_D_max << "\nacc_timer " << accumulator_timer << "\nc_val " << c_val << std::endl;

    // helper lambda to compute par / tree_par strings for logging
    auto par_str = [&]() -> std::string {
        std::ostringstream s;
        s << "parallax " << (parallax_num ? parallax_sum / parallax_num : 0.0)
          << " tree parallax " << (parallax_tree_num ? parallax_tree_sum / parallax_tree_num : 0.0);
        return s.str();
    };
    auto stats_str = [&](const char *label) -> std::string {
        std::ostringstream s;
        s << label
          << " frame=" << frame_count
          << " last_trk=" << last_track_num << " long_trk=" << long_track_num
          << " new_feat=" << filtered_new_feature_num
          << " last_t=" << last_t_track_num << " long_t=" << long_t_track_num
          << " new_t=" << filtered_new_t_feature_num
          << " filt_last=" << filtered_last_track_num << " filt_long=" << filtered_long_track_num
          << " filt_last_t=" << filtered_last_t_track_num << " filt_long_t=" << filtered_long_t_track_num
          << " plateau=" << last_track_num_plateau << " acc=" << accumulator_timer
          << " var_thr=" << var_threshold << " fast=" << fast_flag << " var=" << var_flag;
        return s.str();
    };

    oss2 << "=========================================================================\nFM keyframe selection at frame count " << frame_count << std::endl;
    if ((frame_count < 5) || (accumulator_timer > accumulator_timer_thresh))
    {
        oss2 << stats_str("Keyframe(accumulator)") << std::endl << par_str() << std::endl;
        logMessage(oss2.str());
        ///// LOG /////
        accumulator_timer = 0;
        return true;
    }

    if (parallax_num + parallax_tree_num == 0)
    {
        oss2 << "Keyframe: no features for parallax evaluation" << std::endl;
        logMessage(oss2.str());
        ///// LOG /////
        accumulator_timer = 0;
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH;

        bool is_keyframe = (FOCAL_LENGTH * (parallax_sum / parallax_num)) + (parallax_tree_num ? parallax_tree_sum / parallax_tree_num : 0.0) >= 300;
        oss2 << stats_str(is_keyframe ? "Keyframe(parallax)" : "No Keyframe") << std::endl << par_str() << std::endl;
        logMessage(oss2.str());
        ///// LOG /////
        if (is_keyframe)
            accumulator_timer = 0;
        return is_keyframe;
    }
}

bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0; // number of feature occured in this frame and last frame
    last_track_num = 0; // number of features that has been already found in the previous frames
    last_average_parallax = 0;
    new_feature_num = 0; // number of new feature encountered  with the current frame
    long_track_num = 0; // number of features that has been already encountered in the previous frames, for at least 4 frames
    for (auto &id_pts : image) // for every point in the image
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td); // add the point (with x, y, z, p_u, p_v, v_x, v_y) in the f_per_frame variable together with the time offset td
        assert(id_pts.second[0].first == 0); // meaning the camera id is 0 (first camera)
        if(id_pts.second.size() == 2) // meaning that there are two camera ids and so stereocamera case
        {
            f_per_fra.rightObservation(id_pts.second[1].second); // add the points (obtained from the second camera) in the f_per_frame variable with the time offset td
            assert(id_pts.second[1].first == 1); // meaning the camera id is 1 (second camera)
        }

        int feature_id = id_pts.first; // get the feature_id (same as the one received from the message)
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it) 
                          {
            return it.feature_id == feature_id;
                          }); // search in the feature vector, form the beginning to the end if the feature id is present, it return a pointer to the element in the feature vector that has the same feature_id, or if no such element exists it return feature.end()

        if (it == feature.end()) // if it's the first time we encounter a feature with that feature_id
        {
            feature.push_back(FeaturePerId(feature_id, frame_count)); // add the feature_id in the feature vector, with the associated frame_count (counting the first frame in which we saw this feature)
            feature.back().feature_per_frame.push_back(f_per_fra); // in this feature, under the feature_per_frame field add the point data (x, y, z, p_u, p_v, v_x, v_y)
            new_feature_num++; // increase the number of new feature encountered  with the current frame
        }
        else if (it->feature_id == feature_id) // if the feature was already in the feature vector
        {
            it->feature_per_frame.push_back(f_per_fra); // in this feature, under the feature_per_frame field add the point data (x, y, z, p_u, p_v, v_x, v_y) obtained in the current frame
            last_track_num++; // increase the number of features that has been already found in the previous frames
            if( it-> feature_per_frame.size() >= 4) // if the feature has been encounter in at least 4 previous frames
                long_track_num++; // increase the number of features that has been already encountered in the previous frames, for at least 4 frames
        }
    }

    //if (frame_count < 2 || last_track_num < 20)
    //if (frame_count < 2 || last_track_num < 20 || new_feature_num > 0.5 * last_track_num)
    if (frame_count < 2 || last_track_num < 20 || long_track_num < 40 || new_feature_num > 0.5 * last_track_num) // if we are at a frame smaller then the third, or with the current frame we encountered less then 20 features already encountered in teh past, of with the current frame we encountered less then 40 features already encountered in the past for at least 4 times, if we encountered new feature twice the number of the already encountered features return TRUE (keyframe)
        return true;

    for (auto &it_per_id : feature) // for all the feature already saved, together with the newly discovered with this frame
    {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1) // if the feature first appearence is farthest then 2 frames from the current one and [the sum of the first appearence of the feature (in frames) plus the number of time we ennocuntered it is bigger then the actual frame count] = if the feature appeared the first time at least in the previous frame and it also appeared in the current frame
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count); // get the distance of the feature in the image between the last two frames in whic it appeared
            parallax_num++; // increase the number of feature occured in this frame and last frame
        }
    }

    if (parallax_num == 0) // if none of the features appeared in this frame and last frame
    {
        return true; // return true (keyframe)
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH;
        return parallax_sum / parallax_num >= MIN_PARALLAX; // return true if the average parallax is bigger then a threshold
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature) // for all the features
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) // if the feature first appearence is before frame_count_l and the last appearence is after frame_count_r
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point; // get the undistorted feature position at the frame_count_l frame

            b = it.feature_per_frame[idx_r].point; // get the undistorted feature position at the frame_count_r frame
            
            corres.push_back(make_pair(a, b)); // add it to corres
        }
    }
    return corres; 
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::set_tree_Depth(const VectorXd &x)
{
    int t_feature_index = -1;
    for (auto &model_tree : t_feature)
        for (int v = 0; v < (int)boost::num_vertices(model_tree); ++v)
        {
            ModelNode &node = model_tree[v];
            node.used_num = node.tree_per_frame.size();
            if (node.used_num < 3)
                continue;
            node.estimated_depth = x(++t_feature_index);
            node.solve_flag = (node.estimated_depth < 0) ? 2 : 1;
        }
}

void FeatureManager::removeFailures()
{
    ///// LOG /////
    // std::ostringstream oss;
    // oss << "=========================================================================\nFM remove failures before\nnormal features: " << std::endl;
    // for (auto &it_per_id : feature) // for all the visual features
    // {
    //     oss << "    feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " estimated depth " << it_per_id.estimated_depth << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
    //     {  
    //         oss << "        L pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " R pos " << it_per_frame.pointRight.x() << " " << it_per_frame.pointRight.y() << " " << it_per_frame.pointRight.z() << std::endl;
    //     }
    // }
    // oss << "tree features:" << std::endl;
    // for (auto &it_per_id : t_feature) // for all the visual features
    // {
    //     oss << "    tree feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame  << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
    //     {  
    //         oss << "        pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << std::endl;
    //     }
    // }
    // logMessage(oss.str());
    ///// LOG /////
    double normal_features_removed = 0;
    double tree_features_removed = 0;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) // for all the features, init two iterator pointing to the first and second feature 
    {
        it_next++;
        if (it->solve_flag == 2) // if some feature has not been optimied 
        {
            feature.erase(it); // delete them
            normal_features_removed++;
        }
            
    }
    if (USE_TREE)
    {
        for (auto &model_tree : t_feature)
        {
            // Collect failed vertices, highest index first.
            // Processing highest-first keeps all lower indices stable after
            // each remove_vertex call, so no index adjustment is needed.
            std::vector<int> to_remove;
            for (int v = 0; v < (int)boost::num_vertices(model_tree); ++v)
                if (model_tree[v].solve_flag == 2)
                    to_remove.push_back(v);
            std::sort(to_remove.rbegin(), to_remove.rend());

            for (int v : to_remove)
            {
                node_bypass(model_tree, v);
                tree_features_removed++;
                // BGL shifts all indices > v down by 1; remaining to_remove
                // entries are all < v, so no adjustment is needed.
            }
        }
        // Remove trees that became too small to be useful (< 2 nodes).
        t_feature.erase(
            std::remove_if(t_feature.begin(), t_feature.end(),
                           [](const ModelTree &t){ return boost::num_vertices(t) < 2; }),
            t_feature.end());
    }
    ///// LOG /////
    std::ostringstream oss1;
    oss1 << "=========================================================================\nFM remove failures removed\nnormal features " << normal_features_removed << "\ntree features " << tree_features_removed << std::endl;
    // oss1 << "=========================================================================\nFM remove failures after\nnormal features: " << std::endl;
    // for (auto &it_per_id : feature) // for all the visual features
    // {
    //     oss1 << "    feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " estimated depth " << it_per_id.estimated_depth << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
    //     {  
    //         oss1 << "        L pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " R pos " << it_per_frame.pointRight.x() << " " << it_per_frame.pointRight.y() << " " << it_per_frame.pointRight.z() << std::endl;
    //     }
    // }
    // oss1 << "tree features:" << std::endl;
    // for (auto &it_per_id : t_feature) // for all the visual features
    // {
    //     oss1 << "    tree feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame  << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
    //     {  
    //         oss1 << "        pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << std::endl;
    //     }
    // }
    logMessage(oss1.str());
    ///// LOG /////
}

void FeatureManager::clearDepth()
{
    for (auto &it_per_id : feature)
        it_per_id.estimated_depth = -1;

    if (USE_TREE)
        for (auto &model_tree : t_feature)
            for (int v = 0; v < (int)boost::num_vertices(model_tree); ++v)
                model_tree[v].estimated_depth = -1;
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount()); // init the vector long as the number of features
    int feature_index = -1;
    for (auto &it_per_id : feature) // for every feature
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // assign the used_num
        if (it_per_id.used_num < 4) // if smaller then 4 skip the feature
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth; // in any case assign the depth in the vector to 1/depth of the feature in the image
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

VectorXd FeatureManager::get_tree_DepthVector()
{
    VectorXd dep_vec(get_tree_FeatureCount());
    int feature_index = -1;
    for (auto &model_tree : t_feature)
        for (int v = 0; v < (int)boost::num_vertices(model_tree); ++v)
        {
            ModelNode &node = model_tree[v];
            node.used_num = node.tree_per_frame.size();
            if (node.used_num < 3)
                continue;
            dep_vec(++feature_index) = node.estimated_depth;
        }
    return dep_vec;
}


void FeatureManager::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero(); // init the design matrix (projection matrix P) and fil it with the elements 
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>(); // solve with the SVD the triangulation problem obtaining the £D position of the point in homogeneous coordinates
    point_3d(0) = triangulated_point(0) / triangulated_point(3); // evaluate the point position in euclidean space (x)
    point_3d(1) = triangulated_point(1) / triangulated_point(3); // evaluate the point position in euclidean space (y)
    point_3d(2) = triangulated_point(2) / triangulated_point(3); // evaluate the point position in euclidean space (z)
}

Eigen::MatrixXd FeatureManager::buildEpipolarConstraintMatrix( vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D)
{
    size_t n = pts3D.size();

    // Each point gives 2 rows, each row has 12 columns for linear PnP
    Eigen::MatrixXd A(2*n, 12);

    for (size_t i = 0; i < n; ++i)
    {
        double X = pts3D[i].x;
        double Y = pts3D[i].y;
        double Z = pts3D[i].z;
        double u = pts2D[i].x;
        double v = pts2D[i].y;

        // Row for u-coordinate
        A(2*i, 0)  = X;
        A(2*i, 1)  = Y;
        A(2*i, 2)  = Z;
        A(2*i, 3)  = 1.0;
        A(2*i, 4)  = 0.0;
        A(2*i, 5)  = 0.0;
        A(2*i, 6)  = 0.0;
        A(2*i, 7)  = 0.0;
        A(2*i, 8)  = -u * X;
        A(2*i, 9)  = -u * Y;
        A(2*i, 10) = -u * Z;
        A(2*i, 11) = -u;

        // Row for v-coordinate
        A(2*i+1, 0)  = 0.0;
        A(2*i+1, 1)  = 0.0;
        A(2*i+1, 2)  = 0.0;
        A(2*i+1, 3)  = 0.0;
        A(2*i+1, 4)  = X;
        A(2*i+1, 5)  = Y;
        A(2*i+1, 6)  = Z;
        A(2*i+1, 7)  = 1.0;
        A(2*i+1, 8)  = -v * X;
        A(2*i+1, 9)  = -v * Y;
        A(2*i+1, 10) = -v * Z;
        A(2*i+1, 11) = -v;
    }

    return A;

}

bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, 
                                      vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D)
{
    Eigen::Matrix3d R_initial;
    Eigen::Vector3d P_initial;

    // w_T_cam ---> cam_T_w 
    R_initial = R.inverse();
    P_initial = -(R_initial * P);

    //printf("pnp size %d \n",(int)pts2D.size() );
    if (int(pts2D.size()) < 4)
    {
        printf("feature tracking not enough, please slowly move you device! \n");
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r); // transform the rotational matrix in opencv format
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t); // transform the translational vector in opencv format
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);   // get intrinsic camera parameters
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1); // obtain transformation matrix that transform a 3D point in the object coordinate frame to the camera coordinate frame
    //pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 / focalLength, 0.99, inliers);

    if(!pnp_succ)
    {
        printf("pnp failed ! \n");
        return false;
    }
    cv::Rodrigues(rvec, r); // transform the rotation matrix in rodriguez vector
    //cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp); // transform the rodriguez vector in eigen rotational matrix
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp); // transform the translational vector in eigen translational vector

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();
    P = R * (-T_pnp);

    return true;
}

double FeatureManager::computeError(const std::vector<Eigen::Vector3d>& model, const std::vector<Eigen::Vector3d>& scene, const Eigen::Matrix3d& R, const Eigen::Vector3d& t) 
{
    double error = 0.0;
    for (size_t i = 0; i < model.size(); ++i) {
        Eigen::Vector3d transformed = R * model[i] + t;
        error += (scene[i] - transformed).squaredNorm();
    }
    return std::sqrt(error / model.size());
}

double FeatureManager::computeErrorPointToLine(const std::vector<Eigen::Vector3d>& model, const std::vector<Eigen::Vector3d>& scene, const std::vector<Eigen::Vector3d>& model_axes, const Eigen::Matrix3d& R, const Eigen::Vector3d& t) 
{
    double error = 0.0;
    for (size_t i = 0; i < model.size(); ++i) {
        Eigen::Vector3d transformed = R * model[i] + t;
        Eigen::Vector3d diff = scene[i] - transformed;
        // Distance from point to line: ||diff × axis|| / ||axis||
        // Since axis is unit vector: dist = ||diff × axis||
        double dist = diff.cross(model_axes[i]).norm();
        error += dist * dist;
    }
    return std::sqrt(error / model.size());
}

std::pair<std::pair<Eigen::Matrix3d, Eigen::Vector3d>, double> FeatureManager::estimatePoseSVD(const std::vector<Eigen::Vector3d>& model, const std::vector<Eigen::Vector3d>& scene, const std::vector<Vector3d>& model_axes, double& cond_num) 
{
    // variable to store the final result
    Eigen::Matrix3d R_final;
    Eigen::Vector3d t_final;
    double final_error = 1e10;

    if (model.size() != scene.size() || model.size() < 3) {
        ROS_WARN("Error: model and scene must have same size and >= 3 points\n");
        return std::make_pair(std::make_pair(R_final, t_final), final_error);
    }

    if(ICP_P2L) // point to line ICP
    {
        size_t n = model.size();

        // Build linear system: A * x = b
        // where x = [wx, wy, wz, tx, ty, tz] (rotation vector + translation)
        // Each point-to-line constraint gives us 3 equations (one for each component of cross product)
        Eigen::MatrixXd A(3 * n, 6);
        Eigen::VectorXd b(3 * n);

        for (size_t i = 0; i < n; ++i) {
            const Eigen::Vector3d& d = model_axes[i];              // axis direction (unit vector)
            const Eigen::Vector3d& p_model = model[i];             // point on axis
            const Eigen::Vector3d& p_scene = scene[i];             // scene point

            // Compute the constraint vector: (p_model - p_scene) × d
            // We want (transformed p_model - p_scene) × d  0
            Eigen::Vector3d constraint_rhs = (p_model - p_scene).cross(d);

            // For each component of the cross product (3 equations per correspondence)
            for (int j = 0; j < 3; ++j) {
                int row = 3 * i + j;

                // A matrix rows: (d × p_model) × e_j and d × e_j
                // where e_j is the j-th standard basis vector
                // Simplified: we build the Jacobian of (? × p_model + t - p_scene) × d
                // w.r.t. ? and t

                Eigen::Vector3d cross_d_p = d.cross(p_model);
                Eigen::Vector3d cross_d_e(0, 0, 0);
                cross_d_e(j) = 1.0;

                // Jacobian of (? × p_model) × d w.r.t. ?: (d × p_model) × e_j + d × (e_j × p_model)
                // Simplified form: we need the j-th row of [d ×]
                Eigen::Vector3d jac_omega = cross_d_p;  // Approximation
                A(row, 0) = cross_d_p(0);
                A(row, 1) = cross_d_p(1);
                A(row, 2) = cross_d_p(2);

                // Jacobian of t × d w.r.t. t: d × e_j (rotation of d)
                Eigen::Vector3d cross_d_basis = d.cross(cross_d_e);
                A(row, 3) = cross_d_basis(0);
                A(row, 4) = cross_d_basis(1);
                A(row, 5) = cross_d_basis(2);

                // RHS: j-th component of (p_model - p_scene) × d
                b(row) = -constraint_rhs(j);
            }
        }

        // Solve using SVD: find least-squares solution
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, ComputeThinU | ComputeThinV);
        Eigen::VectorXd x = svd.solve(b);

        // evaluate conditional number
        Eigen::Vector3d singular_values = svd.singularValues();
        cond_num = singular_values(0) / singular_values(2);

        // Extract rotation vector (small angle approximation)
        Eigen::Vector3d omega(x(0), x(1), x(2));
        double theta = omega.norm();

        // Compute rotation matrix using Rodrigues' formula
        if (theta > 1e-6) {
            Eigen::Vector3d axis = omega / theta;
            R_final = AngleAxisd(theta, axis).toRotationMatrix();
        } else {
            R_final = Eigen::Matrix3d::Identity();
        }

        // Extract translation
        t_final = Eigen::Vector3d(x(3), x(4), x(5));

        // Compute error
        final_error = computeErrorPointToLine(model, scene, model_axes, R_final, t_final);

    }
    else // point to point ICP
    {
        // Compute centroids
        Eigen::Vector3d model_center = Eigen::Vector3d::Zero();
        Eigen::Vector3d scene_center = Eigen::Vector3d::Zero();

        for (const auto& p : model) model_center += p;
        for (const auto& p : scene) scene_center += p;

        model_center /= model.size();
        scene_center /= scene.size();
        
        // Center the point sets
        std::vector<Eigen::Vector3d> model_centered(model.size());
        std::vector<Eigen::Vector3d> scene_centered(scene.size());

        for (size_t i = 0; i < model.size(); ++i) {
            model_centered[i] = model[i] - model_center;
            scene_centered[i] = scene[i] - scene_center;
        }

        // Build covariance matrix H
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        for (size_t i = 0; i < model.size(); ++i) {
            H += scene_centered[i] * model_centered[i].transpose();
        }
        
        // SVD decomposition: H = U * S * V^T
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, ComputeFullU | ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        // evaluate conditional number
        Eigen::Vector3d singular_values = svd.singularValues();
        cond_num = singular_values(0) / singular_values(2);

        // Compute rotation matrix
        R_final = U * V.transpose();

        // Ensure proper rotation (det(R) = 1, not -1 for reflection)
        if (R_final.determinant() < 0) {
            V.col(2) *= -1;
            R_final = U * V.transpose();
        }
        
        // Compute translation
        t_final = scene_center - R_final * model_center;

        // Compute error
        final_error = computeError(model, scene, R_final, t_final);
    }
    return std::make_pair(std::make_pair(R_final, t_final), final_error);
}

bool FeatureManager::estimatePoseICPPrior(std::vector<Eigen::Vector3d> model, std::vector<Eigen::Vector3d> scene, const std::vector<Vector3d>& model_axes, int max_iterations, double convergence_threshold, Eigen::Matrix3d& R_init, Eigen::Vector3d& t_init, double& cond_num_final) 
{
    if((model.size() != scene.size()) || (model.size() < 3))
    {
        return false;
    }
    // Initialize with provided estimates or identity
    Eigen::Matrix3d R = R_init;
    Eigen::Vector3d t = t_init;
    double prev_error = 1e10;
    double final_error = 1e10;
    double cond_num = 0.0;

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Correspondence step: transform model and find matches
        std::vector<Vector3d> transformed(model.size());
        for (size_t i = 0; i < model.size(); ++i) {
            transformed[i] = R * model[i] + t;
        }

        // Solve SVD with current correspondences
        cond_num = 0.0;
        auto step_result = estimatePoseSVD(transformed, scene, model_axes, cond_num);

        // Incremental update
        Eigen::Matrix3d R_delta = step_result.first.first;
        Eigen::Vector3d t_delta = step_result.first.second;

        R = R_delta * R;
        t = R_delta * t + t_delta;
        final_error = step_result.second;

        // Check convergence
        double error_change = std::abs(prev_error - final_error);
        if (error_change < convergence_threshold) {
            break;
        }

        prev_error = final_error;
    }
    
    R_init = R;
    t_init = t;
    cond_num_final = cond_num;
    return true;
}

// std::pair<std::pair<Eigen::Matrix3d, Eigen::Vector3d>, double> FeatureManager::estimatePoseICPRansac(std::vector<Eigen::Vector3d> model, std::vector<Eigen::Vector3d> scene, const std::vector<Vector3d>& model_axes, int max_iterations, double convergence_threshold) 
// {
//     if (model.size() < 3) {
//         ROS_WARN("Error: need at least 3 points for RANSAC\n");
//         return std::make_pair(std::make_pair(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()), 1e10);
//     }

//     // RANSAC parameters
//     int num_samples = 3;  // minimum samples needed for SVD
//     int ransac_iterations = std::max(100, static_cast<int>(model.size() * 2));
//     double inlier_threshold = 0.1;  // threshold for considering a point as inlier
//     int best_inlier_count = 0;
    
//     Eigen::Matrix3d R_best = Eigen::Matrix3d::Identity();
//     Eigen::Vector3d t_best = Eigen::Vector3d::Zero();
//     double best_error = 1e10;

//     // Random number generator
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dis(0, model.size() - 1);

//     // RANSAC iterations
//     for (int ransac_iter = 0; ransac_iter < ransac_iterations; ++ransac_iter) {
//         // Sample random subset of correspondences
//         std::vector<size_t> sample_indices;
//         std::set<size_t> unique_indices;
        
//         while (unique_indices.size() < num_samples) {
//             size_t idx = dis(gen);
//             unique_indices.insert(idx);
//         }
//         sample_indices.assign(unique_indices.begin(), unique_indices.end());

//         // Build sampled point sets
//         std::vector<Eigen::Vector3d> model_sample(num_samples);
//         std::vector<Eigen::Vector3d> scene_sample(num_samples);
//         std::vector<Eigen::Vector3d> axes_sample(num_samples);

//         for (size_t i = 0; i < num_samples; ++i) {
//             model_sample[i] = model[sample_indices[i]];
//             scene_sample[i] = scene[sample_indices[i]];
//             axes_sample[i] = model_axes[sample_indices[i]];
//         }

//         // Estimate pose from sample using SVD
//         auto sample_result = estimatePoseSVD(model_sample, scene_sample, axes_sample);
//         Eigen::Matrix3d R_sample = sample_result.first.first;
//         Eigen::Vector3d t_sample = sample_result.first.second;

//         // Count inliers: points where transformed model is close to scene
//         int inlier_count = 0;
//         double sample_error = 0.0;

//         for (size_t i = 0; i < model.size(); ++i) {
//             Eigen::Vector3d transformed = R_sample * model[i] + t_sample;
//             double point_error = (transformed - scene[i]).norm();

//             if (point_error < inlier_threshold) {
//                 inlier_count++;
//             }
//             sample_error += point_error;
//         }

//         // Update best hypothesis if this sample has more inliers
//         if (inlier_count > best_inlier_count || 
//             (inlier_count == best_inlier_count && sample_error < best_error)) {
//             best_inlier_count = inlier_count;
//             best_error = sample_error;
//             R_best = R_sample;
//             t_best = t_sample;
//         }
//     }

//     // Refine with all inliers using ICP iterations on best model
//     Eigen::Matrix3d R = R_best;
//     Eigen::Vector3d t = t_best;
//     double prev_error = best_error;
//     double final_error = best_error;

//     for (int iter = 0; iter < max_iterations; ++iter) {
//         // Filter inliers based on current transformation
//         std::vector<Eigen::Vector3d> model_inliers;
//         std::vector<Eigen::Vector3d> scene_inliers;
//         std::vector<Eigen::Vector3d> axes_inliers;

//         for (size_t i = 0; i < model.size(); ++i) {
//             Eigen::Vector3d transformed = R * model[i] + t;
//             double point_error = (transformed - scene[i]).norm();

//             if (point_error < inlier_threshold) {
//                 model_inliers.push_back(model[i]);
//                 scene_inliers.push_back(scene[i]);
//                 axes_inliers.push_back(model_axes[i]);
//             }
//         }

//         // Need minimum 3 points to estimate pose
//         if (model_inliers.size() < 3) {
//             break;
//         }

//         // Refine with inliers
//         auto step_result = estimatePoseSVD(model_inliers, scene_inliers, axes_inliers);
//         Eigen::Matrix3d R_delta = step_result.first.first;
//         Eigen::Vector3d t_delta = step_result.first.second;

//         // Update transformation
//         R = R_delta * R;
//         t = R_delta * t + t_delta;

//         // Compute error on all points
//         final_error = 0.0;
//         for (size_t i = 0; i < model.size(); ++i) {
//             Eigen::Vector3d transformed = R * model[i] + t;
//             final_error += (transformed - scene[i]).norm();
//         }

//         // Check convergence
//         double error_change = std::abs(prev_error - final_error);
//         if (error_change < convergence_threshold) {
//             break;
//         }

//         prev_error = final_error;
//     }

//     return std::make_pair(std::make_pair(R, t), final_error);
// }

void FeatureManager::initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{

    if(frameCnt > 0) // if it's not the first frame?
    {
        vector<cv::Point2f> pts2D;
        vector<cv::Point3f> pts3D;
        for (auto &it_per_id : feature) // for all the features
        {
            if (it_per_id.estimated_depth > 0) // if it's deeper then 0 (meaning i've already traingulated the point)
            {
                int index = frameCnt - it_per_id.start_frame; // get for how many frames the feature was visible
                if((int)it_per_id.feature_per_frame.size() >= index + 1) // if i have more or equal observation of that feature then index, meaing it has been visible for the whole time (for every frame) 
                {
                    Vector3d ptsInCam = ric[0] * (it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth) + tic[0]; // get the feature position in the camera frame
                    Vector3d ptsInWorld = Rs[it_per_id.start_frame] * ptsInCam + Ps[it_per_id.start_frame]; // get the feature position in the world frame

                    cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z()); // change format of the point in its first appearence
                    cv::Point2f point2d(it_per_id.feature_per_frame[index].point.x(), it_per_id.feature_per_frame[index].point.y()); // get the last appearence in the image
                    pts3D.push_back(point3d); // add it to the vector
                    pts2D.push_back(point2d);  // add it to the vector
                }
            }
        }
        //if(!USE_TREE)
        if(true)
        {
            Eigen::Matrix3d RCam;
            Eigen::Vector3d PCam;
            // trans to w_T_cam
            RCam = Rs[frameCnt - 1] * ric[0];
            PCam = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];

            if(solvePoseByPnP(RCam, PCam, pts2D, pts3D))
            {
                // trans to w_T_imu
                Rs[frameCnt] = RCam * ric[0].transpose(); 
                Ps[frameCnt] = -RCam * ric[0].transpose() * tic[0] + PCam;

                //Eigen::Quaterniond Q(Rs[frameCnt]);
                //cout << "frameCnt: " << frameCnt <<  " pnp Q " << Q.w() << " " << Q.vec().transpose() << endl;
                //cout << "frameCnt: " << frameCnt << " pnp P " << Ps[frameCnt].transpose() << endl;
            }
            
        }

        //if(USE_TREE)
        if(false)
        {   
            Eigen::Matrix3d RCam, RCam_icp, RCam_final;
            Eigen::Vector3d PCam, PCam_icp, PCam_final;
            vector<Eigen::Vector3d> old_t_obs;
            vector<Eigen::Vector3d> cur_t_obs;
            vector<Eigen::Vector3d> old_t_normals;

            // run icp p2p or p2l closed form (SVD) to get an initial estimate: 
            // get points correspondences for all the features that apear in the current frame and their first observations
            for (const auto &model_tree : t_feature)
                for (int v = 0; v < (int)boost::num_vertices(model_tree); ++v)
                {
                    const ModelNode &node = model_tree[v];
                    if (!node.has_frame(frameCnt))
                        continue;
                    auto fpf_it = std::find_if(node.tree_per_frame.begin(), node.tree_per_frame.end(),
                                               [frameCnt](const TreePerFrame &fpf){ return fpf.frame == frameCnt; });
                    if ((fpf_it != node.tree_per_frame.end()) && (node.tree_per_frame.size() > 2) && (node.estimated_depth > 0))
                    {
                        double depth = node.estimated_depth;
                        Eigen::Vector3d anchor_pt_uv = node.tree_per_frame[0].point.normalized();
                        Eigen::Vector3d old_t_obsInCam   = ric[0] * (depth * anchor_pt_uv) + tic[0];
                        Eigen::Vector3d old_t_obsInWorld = Rs[node.start_frame] * old_t_obsInCam + Ps[node.start_frame];
                        Eigen::Vector3d old_t_point3d(old_t_obsInWorld.x(), old_t_obsInWorld.y(), old_t_obsInWorld.z());
                        Eigen::Vector3d old_t_norm = Eigen::Vector3d::Zero();
                        if (ICP_P2L)
                        {
                            Eigen::Vector3d old_t_normInCam  = ric[0] * node.tree_per_frame[0].n;
                            Eigen::Vector3d old_t_normInWorld = Rs[node.start_frame] * old_t_normInCam;
                            old_t_norm = old_t_normInWorld.normalized();
                        }
                        Eigen::Vector3d new_t_point3d(fpf_it->point.x(), fpf_it->point.y(), fpf_it->point.z());
                        old_t_obs.push_back(old_t_point3d);
                        cur_t_obs.push_back(new_t_point3d);
                        if (ICP_P2L)
                            old_t_normals.push_back(old_t_norm);
                    }
                }

            // evaluate ICP solution and relative conditional number
            // normal ICP with prior tranformation matrix
            RCam = Rs[frameCnt - 1] * ric[0];
            PCam = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];
            RCam_icp = RCam;
            PCam_icp = PCam;
            double cond_num_icp = 0.0;
            bool icp_result_f = estimatePoseICPPrior(old_t_obs, cur_t_obs, old_t_normals, 20, 1e-3, RCam_icp, PCam_icp, cond_num_icp);
            bool pnp_result_f = solvePoseByPnP(RCam, PCam, pts2D, pts3D);

            // evaluate pnp solution  and icp soliution
            if((pnp_result_f) && (icp_result_f)) // if you got a solution from both pnp and icp
            {   
                // mix the two results
                // evaluate the pnp condition number
                Eigen::MatrixXd A = buildEpipolarConstraintMatrix(pts2D, pts3D);
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
                double cond_num_pnp = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);

                std::cout << "ICP AND PNP: icp cond num " << cond_num_icp << " pnp condition number " << cond_num_pnp << std::endl;

                // for both the methods extract the angle axis decomposition of the rotation matrix and the unit vector and magnitude of the translation vector
                // pnp
                Eigen::AngleAxisd angleAxispnp(RCam);
                double angle_pnp = angleAxispnp.angle(); 
                Eigen::Vector3d rot_axis_pnp = angleAxispnp.axis(); 
                
                Eigen::Vector3d t_unit_pnp = PCam.normalized();
                double t_mag_pnp = PCam.norm();

                // icp
                Eigen::AngleAxisd angleAxisicp(RCam_icp);
                double angle_icp = angleAxisicp.angle(); 
                Eigen::Vector3d rot_axis_icp = angleAxisicp.axis(); 
                
                Eigen::Vector3d t_unit_icp = PCam_icp.normalized();
                double t_mag_icp = PCam_icp.norm();

                double icp_coef = 0.0;
                double pnp_coef = 0.0;
                // considerations: bigger the conditional number less weight we should attribute to that component
                if(cond_num_pnp > cond_num_icp) // if the pnp algorithm is worst posed then the icp one
                {
                    // evaluate proportional coeficents
                    pnp_coef = (cond_num_icp / cond_num_pnp);
                    icp_coef = 1 - icp_coef;
                    
                }
                else if(cond_num_icp > cond_num_pnp) // if the icp algorithm is worst posed then the pnp one
                {
                    // evaluate proportional coeficents
                    icp_coef = (cond_num_pnp / cond_num_icp);
                    pnp_coef = 1 - pnp_coef;
                    

                }
                else if((cond_num_pnp < 1e6) && (cond_num_icp < 1e6)) // if both the problems are not hill posed
                {
                    // evaluate proportional coeficents
                    pnp_coef = 0.5;
                    icp_coef = 0.5;
                }
                else // both the problems are hill posed
                {
                    return;
                }

                // mix the rotation in a weighted sum
                Eigen::Vector3d final_rot_axis = (icp_coef * rot_axis_icp + pnp_coef * rot_axis_pnp).normalized();
                double final_angle = icp_coef * angle_icp + pnp_coef * angle_pnp;
                RCam_final = Eigen::AngleAxisd(final_angle, final_rot_axis).toRotationMatrix();

                // mix the translation vector
                Eigen::Vector3d final_t_unit = (icp_coef * t_unit_icp + pnp_coef * t_unit_pnp).normalized();
                double final_t_mag = icp_coef * t_mag_pnp + pnp_coef * t_mag_pnp;
                PCam_final = final_t_mag * final_t_unit;
                
                // trans to w_T_imu
                Rs[frameCnt] = RCam_final * ric[0].transpose(); 
                Ps[frameCnt] = -RCam_final * ric[0].transpose() * tic[0] + PCam_final;

                std::cout << "ICP and PNP: R " << Rs[frameCnt] << " t " << Ps[frameCnt] << std::endl;
                // double anglepnp = 1.57;
                // Eigen::Matrix3d RCamtestpnp;
                // RCamtestpnp << 1, 0, 0,
                //         0, 1, 0,
                //         0, 0, 1;

                // double angleicp = 1.57;
                // Eigen::Matrix3d RCamtesticp;
                // RCamtesticp << 1, 0, 0,
                //         0, 1, 0,
                //         0, 0, 1;

                
            }
            else if((!pnp_result_f) && (icp_result_f)) // if you solved only icp
            {
                // trans to w_T_imu
                Rs[frameCnt] = RCam_icp * ric[0].transpose(); 
                Ps[frameCnt] = -RCam_icp * ric[0].transpose() * tic[0] + PCam_icp;

                std::cout << "ICP ONLY: R " << Rs[frameCnt] << " t " << Ps[frameCnt] << std::endl;
            }
            else if((pnp_result_f) && (!icp_result_f)) // if you solved only pnp
            {
                // trans to w_T_imu
                Rs[frameCnt] = RCam * ric[0].transpose(); 
                Ps[frameCnt] = -RCam * ric[0].transpose() * tic[0] + PCam;

                std::cout << "PNP ONLY: R " << Rs[frameCnt] << " t " << Ps[frameCnt] << std::endl;
            } 
            else
            {
                std::cout << "NOR PNP NEITHER ICP" << std::endl;
            }
        }
    }
}

void FeatureManager::triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature) // for all the features
    {
        if (it_per_id.estimated_depth > 0) // if it has already a valid estimated depth, go to the next feature 
            continue;

        if(STEREO && it_per_id.feature_per_frame[0].is_stereo) // if we are in the stereo case
        {
            int imu_i = it_per_id.start_frame; // get the frame count of the first appearence of the feature
            Eigen::Matrix<double, 3, 4> leftPose;
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0]; // get the position at time 0 of the left camera
            Eigen::Matrix3d R0 = Rs[imu_i] * ric[0]; // get the orientation at time 0 of the left camera
            leftPose.leftCols<3>() = R0.transpose(); // save the inverce orientation matrix in the variable
            leftPose.rightCols<1>() = -R0.transpose() * t0; // save the position vector in the opposite frame
            //cout << "left pose " << leftPose << endl;

            Eigen::Matrix<double, 3, 4> rightPose;
            Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[1]; // get the position at time 0 of the right camera
            Eigen::Matrix3d R1 = Rs[imu_i] * ric[1]; // get the orientation at time 0 of the right camera
            rightPose.leftCols<3>() = R1.transpose(); // save the inverce orientation matrix in the variable
            rightPose.rightCols<1>() = -R1.transpose() * t1; // save the position vector in the opposite frame
            //cout << "right pose " << rightPose << endl;

            Eigen::Vector2d point0, point1;
            Eigen::Vector3d point3d;
            point0 = it_per_id.feature_per_frame[0].point.head(2); // get the initial image position of the feature in its first appearence in the left camera
            point1 = it_per_id.feature_per_frame[0].pointRight.head(2); // get the initial image position of the feature in its first appearence in the right camera
            //cout << "point0 " << point0.transpose() << endl;
            //cout << "point1 " << point1.transpose() << endl;

            triangulatePoint(leftPose, rightPose, point0, point1, point3d); // given the left and right observation of the feature evaluate its 3D position in euclidean space
            Eigen::Vector3d localPoint;
            localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>(); // transform the 3D point from world frame to left camera frame
            double depth = localPoint.z(); // get its depth
            if (depth > 0) // if it's >0 save it in the estimated depth term of the feature (if it's smaller then 0 there has been some error in the evaluation)
                it_per_id.estimated_depth = depth;
            else // else init it to INIT_DEPTH (that sould be 0 ?)
                it_per_id.estimated_depth = INIT_DEPTH;
            /*
            Vector3d ptsGt = pts_gt[it_per_id.feature_id];
            printf("stereo %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                            ptsGt.x(), ptsGt.y(), ptsGt.z());
            */
            continue; // depth evaluated so go to th enext feature
        }
        else if(it_per_id.feature_per_frame.size() > 1) // else if we are not in the stereo case and we have more then one observation of the given feature 
        {
            // do the same procedure but instead of triangluating the feature given the two observations in the two cameras, do that with the successive appearence of the feature in time
            int imu_i = it_per_id.start_frame;
            Eigen::Matrix<double, 3, 4> leftPose;
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
            leftPose.leftCols<3>() = R0.transpose();
            leftPose.rightCols<1>() = -R0.transpose() * t0;

            imu_i++;
            Eigen::Matrix<double, 3, 4> rightPose;
            Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_i] * ric[0];
            rightPose.leftCols<3>() = R1.transpose();
            rightPose.rightCols<1>() = -R1.transpose() * t1;

            Eigen::Vector2d point0, point1;
            Eigen::Vector3d point3d;
            point0 = it_per_id.feature_per_frame[0].point.head(2);
            point1 = it_per_id.feature_per_frame[1].point.head(2);
            triangulatePoint(leftPose, rightPose, point0, point1, point3d);
            Eigen::Vector3d localPoint;
            localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
            double depth = localPoint.z();
            if (depth > 0)
                it_per_id.estimated_depth = depth;
            else
                it_per_id.estimated_depth = INIT_DEPTH;
            /*
            Vector3d ptsGt = pts_gt[it_per_id.feature_id];
            printf("motion  %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                            ptsGt.x(), ptsGt.y(), ptsGt.z());
            */
            continue; // depth evaluated so go to th enext feature
        }
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // save the number of appearence of the feature in the used_num attribute
        if (it_per_id.used_num < 4) // if there are less then 4 observations of the feature skip and go to the next one
            continue;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1; // initialize the two variable

        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4); // init the matrix svd_a with the given sizes (number of observations of the features times 4)
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0; // initialize the projection matrix
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0]; // get the position of the feature at time imu_i
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0]; // get the orientation of the feature at time imu_i
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observation of the feature
        {
            imu_j++; // increase imu_j counter

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0]; // get the postion of the feature at frame count = imu_j
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0]; // get the orientation of the feature at frame count = imu_j
            Eigen::Vector3d t = R0.transpose() * (t1 - t0); // get the relative displacements
            Eigen::Matrix3d R = R0.transpose() * R1; // get the relative orientation
            Eigen::Matrix<double, 3, 4> P; // compute the projection matrix for the current frame
            P.leftCols<3>() = R.transpose(); // invert the reference frame (orientation)
            P.rightCols<1>() = -R.transpose() * t; // invert the reference frame (translation)
            Eigen::Vector3d f = it_per_frame.point.normalized(); // get the feature undistorted image position
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0); // add the epipolar constraint for the given observation of the feature
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j) // if the current frame is the same as the initial one skip evaluation since they are not needed
                continue;
        }
        assert(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>(); // perform the SVD obtaining the position of the point in homogeneous coordinate
        double svd_method = svd_V[2] / svd_V[3]; // compute the depth
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method; // assign it to the estimated depth
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1) // in case of small or negative depth (considered invalid)
        {
            it_per_id.estimated_depth = INIT_DEPTH; // assign as depth the INIT_DEPTH value
        }

    }

    if (USE_TREE)
    {
        for (auto &model_tree : t_feature)
            for (int v = 0; v < (int)boost::num_vertices(model_tree); ++v)
            {
                ModelNode &node = model_tree[v];
                if (node.estimated_depth > 0)
                    continue;
                if (node.tree_per_frame.empty())
                    continue;
                double depth = node.tree_per_frame[0].point.norm();
                node.estimated_depth = (depth > 0) ? depth : INIT_DEPTH;
            }
    }
}

void FeatureManager::removeOutlier(set<int> &outlierIndex, set<int> &tree_outlierIndex)
{   
    ///// LOG /////
    std::ostringstream oss;
    oss << "=========================================================================\nFM ro removing outliers\nnormal features: " << std::endl;

    std::set<int>::iterator itSet;
    for (auto it = feature.begin(), it_next = feature.begin(); // for all the features (but creating two pointer to the features)
         it != feature.end(); it = it_next) // 
    {
        it_next++; // get the pointer to the next feature ( in this way it will be a fof loop with it the i feature and it_next the i+1 feature)
        int index = it->feature_id; // get the feature id
        itSet = outlierIndex.find(index); 
        if(itSet != outlierIndex.end()) // if it's in the outlier list
        {
            feature.erase(it); // remove it
            oss << "    removing " << index << std::endl;
            //printf("remove outlier %d \n", index);
        }
    }
    oss << "tree features:" << std::endl;
    if (USE_TREE)
    {
        for (auto &model_tree : t_feature)
        {
            vector<int> to_remove;
            for (int v = 0; v < (int)boost::num_vertices(model_tree); ++v)
            {
                if (tree_outlierIndex.count(model_tree[v].feature_id))
                {
                    oss << "    removing " << model_tree[v].feature_id << std::endl;
                    to_remove.push_back(v);
                }
            }
            for (int i = (int)to_remove.size() - 1; i >= 0; --i)
                node_bypass(model_tree, to_remove[i]);
        }
        t_feature.erase(
            std::remove_if(t_feature.begin(), t_feature.end(),
                           [](const ModelTree &t){ return boost::num_vertices(t) < 2; }),
            t_feature.end());
    }
    logMessage(oss.str());
    ///// LOG /////
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{   
    ///// LOG /////
    // std::ostringstream oss;
    // oss << "=========================================================================\nFM remove back shift depth before\nnormal features: " << std::endl;
    // for (auto &it_per_id : feature) // for all the visual features
    // {
    //     oss << "    feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " estimated depth " << it_per_id.estimated_depth << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
    //     {  
    //         oss << "        L pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " R pos " << it_per_frame.pointRight.x() << " " << it_per_frame.pointRight.y() << " " << it_per_frame.pointRight.z() << std::endl;
    //     }
    // }
    // oss << "tree features:" << std::endl;
    // for (auto &it_per_id : t_feature) // for all the visual features
    // {
    //     oss << "    tree feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame  << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
    //     {  
    //         oss << "        pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << std::endl;
    //     }
    // }
    // logMessage(oss.str());
    ///// LOG /////
    int normal_features_removed = 0;
    int tree_features_removed = 0;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) // for all the features, init two iterator pointing to the first and second feature 
    {
        it_next++;

        if (it->start_frame != 0) // if the first feature first appearence is different from 0 decrease it
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point; // get the oldest position observed of the feature
            it->feature_per_frame.erase(it->feature_per_frame.begin()); // delete the first observation
            if (it->feature_per_frame.size() < 2) // if you have less then 2 observation
            {
                normal_features_removed++;
                feature.erase(it); // delete the feature and process the next one
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth; // get the feature 3d position
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P; // project it in the new frame (which one?)
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P); // project it in the camera frame (i guess?)
                double dep_j = pts_j(2); // re-evaluate the depth
                if (dep_j > 0) // if valid assign it to the estimated depth
                    it->estimated_depth = dep_j;
                else // otherwise give it the INIT_DEPTH value
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }

    if (USE_TREE)
    {
        for (auto &model_tree : t_feature)
        {
            vector<int> to_remove;
            for (int v = 0; v < (int)boost::num_vertices(model_tree); ++v)
            {
                ModelNode &node = model_tree[v];
                if (node.start_frame != 0)
                {
                    node.start_frame--;
                    for (auto &fpf : node.tree_per_frame) fpf.frame--;
                }
                else
                {
                    Eigen::Vector3d anchor_pt = node.tree_per_frame[0].point.normalized();
                    node.tree_per_frame.erase(node.tree_per_frame.begin());
                    for (auto &fpf : node.tree_per_frame) fpf.frame--;

                    if ((int)node.tree_per_frame.size() < 2)
                    {
                        to_remove.push_back(v);
                        tree_features_removed++;
                    }
                    else
                    {
                        Eigen::Vector3d pts_i   = anchor_pt * node.estimated_depth;
                        Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                        Eigen::Vector3d pts_j   = new_R.transpose() * (w_pts_i - new_P);
                        double dep_j = pts_j(2);
                        node.estimated_depth = (dep_j > 0) ? dep_j : INIT_DEPTH;
                        node.start_frame = node.tree_per_frame[0].frame;
                    }
                }
            }
            for (int i = (int)to_remove.size() - 1; i >= 0; --i)
                node_bypass(model_tree, to_remove[i]);
        }
        t_feature.erase(
            std::remove_if(t_feature.begin(), t_feature.end(),
                           [](const ModelTree &t){ return boost::num_vertices(t) < 2; }),
            t_feature.end());
    }

    ///// LOG /////
    std::ostringstream oss1;
    oss1 << "=========================================================================\nFM remove back shift depth removed\nnormal features " <<  normal_features_removed << "\ntree features " << tree_features_removed << std::endl;
    // oss1 << "-------------------------------------------------------------------------\nFM remove back shift depth after\nnormal features: " << std::endl;
    // for (auto &it_per_id : feature) // for all the visual features
    // {
    //     oss1 << "    feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " estimated depth " << it_per_id.estimated_depth << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
    //     {  
    //         oss1 << "        L pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " R pos " << it_per_frame.pointRight.x() << " " << it_per_frame.pointRight.y() << " " << it_per_frame.pointRight.z() << std::endl;
    //     }
    // }
    // oss1 << "tree features:" << std::endl;
    // for (auto &it_per_id : t_feature) // for all the visual features
    // {
    //     oss1 << "    tree feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame  << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
    //     {  
    //         oss1 << "        pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << std::endl;
    //     }
    // }
    logMessage(oss1.str());
    ///// LOG /////
}

void FeatureManager::removeBack()
{
    ///// LOG /////
    // std::ostringstream oss;
    // oss << "=========================================================================\nFM remove back before\nnormal features: " << std::endl;
    // for (auto &it_per_id : feature) // for all the visual features
    // {
    //     oss << "    feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " estimated depth " << it_per_id.estimated_depth << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
    //     {  
    //         oss << "        L pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " R pos " << it_per_frame.pointRight.x() << " " << it_per_frame.pointRight.y() << " " << it_per_frame.pointRight.z() << std::endl;
    //     }
    // }
    // oss << "tree features:" << std::endl;
    // for (auto &it_per_id : t_feature) // for all the visual features
    // {
    //     oss << "    tree feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame  << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
    //     {  
    //         oss << "        pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << std::endl;
    //     }
    // }
    // logMessage(oss.str());
    ///// LOG /////
    int normal_features_removed = 0;
    int tree_features_removed = 0;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) // for all the features, init two iterator pointing to the first and second feature 
    {
        it_next++;

        if (it->start_frame != 0) // if the first feature first appearence is different from 0 decrease it
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin()); // delete the first appearence of the feature
            if (it->feature_per_frame.size() == 0) // if you don't have observations of the feature
            {
                normal_features_removed++;
                feature.erase(it); // delete the whole feature
            }
        }
    }

    if (USE_TREE)
    {
        for (auto &model_tree : t_feature)
        {
            vector<int> to_remove;
            for (int v = 0; v < (int)boost::num_vertices(model_tree); ++v)
            {
                ModelNode &node = model_tree[v];
                if (node.start_frame != 0)
                {
                    node.start_frame--;
                    for (auto &fpf : node.tree_per_frame) fpf.frame--;
                }
                else
                {
                    node.tree_per_frame.erase(node.tree_per_frame.begin());
                    for (auto &fpf : node.tree_per_frame) fpf.frame--;
                    if ((int)node.tree_per_frame.size() < 2)
                    {
                        to_remove.push_back(v);
                        tree_features_removed++;
                    }
                    else
                        node.start_frame = node.tree_per_frame[0].frame;
                }
            }
            for (int i = (int)to_remove.size() - 1; i >= 0; --i)
                node_bypass(model_tree, to_remove[i]);
        }
        t_feature.erase(
            std::remove_if(t_feature.begin(), t_feature.end(),
                           [](const ModelTree &t){ return boost::num_vertices(t) < 2; }),
            t_feature.end());
    }

    ///// LOG /////
    std::ostringstream oss1;
    oss1 << "=========================================================================\nFM remove back removed\nnormal features " <<  normal_features_removed << "\ntree features " << tree_features_removed << std::endl;
    // oss1 << "-------------------------------------------------------------------------\nFM remove back after\nnormal features: " << std::endl;
    // for (auto &it_per_id : feature) // for all the visual features
    // {
    //     oss1 << "    feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " estimated depth " << it_per_id.estimated_depth << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
    //     {  
    //         oss1 << "        L pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " R pos " << it_per_frame.pointRight.x() << " " << it_per_frame.pointRight.y() << " " << it_per_frame.pointRight.z() << std::endl;
    //     }
    // }
    // oss1 << "tree features:" << std::endl;
    // for (auto &it_per_id : t_feature) // for all the visual features
    // {
    //     oss1 << "    tree feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame  << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
    //     {  
    //         oss1 << "        pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << std::endl;
    //     }
    // }
    // logMessage(oss1.str());
    ///// LOG /////
}

void FeatureManager::removeFront(int frame_count)
{
    ///// LOG /////
    // std::ostringstream oss;
    // oss << "=========================================================================\nFM remove front before\nnormal features: " << std::endl;
    // for (auto &it_per_id : feature) // for all the visual features
    // {
    //     oss << "    feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " estimated depth " << it_per_id.estimated_depth << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
    //     {  
    //         oss << "        L pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " R pos " << it_per_frame.pointRight.x() << " " << it_per_frame.pointRight.y() << " " << it_per_frame.pointRight.z() << std::endl;
    //     }
    // }
    // oss << "tree features:" << std::endl;
    // for (auto &it_per_id : t_feature) // for all the visual features
    // {
    //     oss << "    tree feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame  << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
    //     {  
    //         oss << "        pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << std::endl;
    //     }
    // }
    // logMessage(oss.str());
    ///// LOG /////
    int normal_features_removed = 0;
    int tree_features_removed = 0;
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) // for all the features, init two iterator pointing to the first and second feature 
    {
        it_next++;

        if (it->start_frame == frame_count) // if the start frame is equal to the frame count 
        {
            it->start_frame--; // decrease it
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame; // get the interval between the first apperence of the feature and the first element of the window
            if (it->endFrame() < frame_count - 1) // if the feature last frame apperence is smaller then the second to last frame count ? skip it
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j); // otherwise erase the feature observation 
            if (it->feature_per_frame.size() == 0) // if you no longer have observations of the feature
            {
                feature.erase(it); // delete it
                normal_features_removed++;
            }
        }
    }
    vector<int> removed_id;
    if (USE_TREE)
    {
        for (auto &model_tree : t_feature)
        {
            vector<int> to_remove;
            for (int v = 0; v < (int)boost::num_vertices(model_tree); ++v)
            {
                ModelNode &node = model_tree[v];
                if (node.start_frame == frame_count)
                {
                    node.start_frame--;
                    for (auto &fpf : node.tree_per_frame) fpf.frame--;
                }
                else
                {
                    if (node.endFrame() < frame_count - 1)
                        continue;
                    if (node.has_frame(WINDOW_SIZE - 1))
                    {
                        auto fpf_it = std::find_if(node.tree_per_frame.begin(), node.tree_per_frame.end(),
                                                   [](const TreePerFrame &item){ return item.frame == WINDOW_SIZE - 1; });
                        if (fpf_it != node.tree_per_frame.end())
                            node.tree_per_frame.erase(fpf_it);
                    }
                    for (auto &obs : node.tree_per_frame)
                        if (obs.frame > WINDOW_SIZE - 1) obs.frame -= 1;

                    if (node.tree_per_frame.empty())
                    {
                        removed_id.push_back(node.feature_id);
                        to_remove.push_back(v);
                        tree_features_removed++;
                    }
                    else if (node.start_frame != node.tree_per_frame[0].frame)
                    {
                        std::cout << "REMOVE FRONT ERROR prev start frame " << node.start_frame
                                  << " new one " << node.tree_per_frame[0].frame << std::endl;
                        node.start_frame = node.tree_per_frame[0].frame;
                    }
                }
            }
            for (int i = (int)to_remove.size() - 1; i >= 0; --i)
                node_bypass(model_tree, to_remove[i]);
        }
        t_feature.erase(
            std::remove_if(t_feature.begin(), t_feature.end(),
                           [](const ModelTree &t){ return boost::num_vertices(t) < 2; }),
            t_feature.end());
    }
    ///// LOG /////
    std::ostringstream oss1;
    // oss1 << "-------------------------------------------------------------------------\nFM tree features removed " << std::endl;
    // for(const auto& r_id : removed_id)
    // {
    //     oss1 << r_id << std::endl;
    // }
    oss1 << "-------------------------------------------------------------------------\nFM remove front removed\nnormal features " << normal_features_removed << "\ntree features " << tree_features_removed << std::endl;
    
    // oss1 << "-------------------------------------------------------------------------\nFM remove front after\nnormal features: " << std::endl;
    // for (auto &it_per_id : feature) // for all the visual features
    // {
    //     oss1 << "    feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " estimated depth " << it_per_id.estimated_depth << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
    //     {  
    //         oss1 << "        L pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " R pos " << it_per_frame.pointRight.x() << " " << it_per_frame.pointRight.y() << " " << it_per_frame.pointRight.z() << std::endl;
    //     }
    // }
    // oss1 << "tree features:" << std::endl;
    // for (auto &it_per_id : t_feature) // for all the visual features
    // {
    //     oss1 << "    tree feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame  << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
    //     {  
    //         oss1 << "        pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << std::endl;
    //     }
    // }
    logMessage(oss1.str());
    ///// LOG /////
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame]; // get the feature informations (x, y, x, p_u, p_v, v_x, v_y) of the time before the last time we encountered the feature
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame]; // get the feature informations (x, y, x, p_u, p_v, v_x, v_y) of the last time we encountered the feature (would be current frame when teh function is called)

    double ans = 0;
    Vector3d p_j = frame_j.point; // get the x, y, z of j

    double u_j = p_j(0); // get the undistorted x
    double v_j = p_j(1); // get the undistorted y

    Vector3d p_i = frame_i.point; // get the x, y, z of i
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2); // get the z of i
    double u_i = p_i(0) / dep_i; // normalize
    double v_i = p_i(1) / dep_i; // normalize
    double du = u_i - u_j, dv = v_i - v_j; // evaluate the difference in position of the feature point in the image between last two apperence of the feature

    double dep_i_comp = p_i_comp(2); // same as above ?
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp))); // get minimum distance in the image of the feature point in the last two frame in which it appears

    return ans;
}

double FeatureManager::compensated_tree_Parallax2(const ModelNode &node, int frame_count)
{
    // build T_tree_lcam from T_lcam_tree
    Eigen::Matrix3d R_lcam_tree = T_lcam_tree.block<3, 3>(0, 0);
    Eigen::Vector3d P_lcam_tree = T_lcam_tree.block<3, 1>(0, 3);
    Eigen::Matrix3d R_tree_lcam = R_lcam_tree.transpose();
    Eigen::Vector3d P_tree_lcam = -R_tree_lcam * P_lcam_tree;
    Eigen::Matrix4d T_tree_lcam;
    T_tree_lcam << R_tree_lcam, P_tree_lcam, Eigen::RowVector4d(0, 0, 0, 1);

    Eigen::VectorXd K_vec(9);
    K_vec << ref_frame["tree_camera"].k[0], 0.0, ref_frame["tree_camera"].k[2],
             0.0, ref_frame["tree_camera"].k[4], ref_frame["tree_camera"].k[5],
             0.0, 0.0, 1.0;
    Eigen::Matrix3d K_mat = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(K_vec.data());

    // project a 3-D point (in lcam frame) to image pixel using T_lcam_tree and K
    auto project = [&](const TreePerFrame &obs) -> Eigen::Vector2d {
        Eigen::Vector4d p_h;
        p_h << obs.point.x(), obs.point.y(), obs.point.z(), 1.0;
        Eigen::Vector3d p_cam = (T_lcam_tree * p_h).head<3>();
        Eigen::Vector3d p_uv  = K_mat * p_cam;
        return Eigen::Vector2d(p_uv[0] / p_uv[2], p_uv[1] / p_uv[2]);
    };

    // parallax between the observation at frame_count-2 and frame_count-1
    Eigen::Vector2d p_img_i = Eigen::Vector2d::Zero();
    Eigen::Vector2d p_img_j = Eigen::Vector2d::Zero();
    for (const auto &obs : node.tree_per_frame)
    {
        if (obs.frame == frame_count - 2) p_img_i = project(obs);
        else if (obs.frame == frame_count - 1) p_img_j = project(obs);
    }

    double du = p_img_i[0] - p_img_j[0];
    double dv = p_img_i[1] - p_img_j[1];
    return std::sqrt(du * du + dv * dv);
}