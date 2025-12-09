#pragma once

#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include "utils/common_utils.h"
#include "SplineState.h"
#include "ikd-Tree/ikd_Tree.h"

class Association
{
public:

    template<class PointType>
    static void pointBodyToWorld(int64_t t_ns, const SplineState* spline, const PointType& pi, PointType& po, const Eigen::Vector3d& t_bl, const Eigen::Quaterniond& q_bl) 
    {    
        Eigen::Quaterniond q;
        Eigen::Vector3d pos = spline->itpPosition(t_ns);
        spline->itpQuaternion(t_ns, &q);
        Eigen::Vector3f p_body(pi.x, pi.y, pi.z);
        Eigen::Vector3f p_global = q.cast<float>() * (q_bl.cast<float>() * p_body + t_bl.cast<float>()) + pos.cast<float>();
        po.x = p_global(0);
        po.y = p_global(1);
        po.z = p_global(2);
        po.curvature = pi.curvature;
    }     

    static void findCorresp(int& effect_num_k, const SplineState* spline, KD_TREE<pcl::PointXYZINormal>* ikdtree, Eigen::aligned_deque<PointData>& pt_meas)
    {
        int num_pt = pt_meas.size();
        #pragma omp parallel for num_threads(NUM_OF_THREAD) schedule(dynamic)
        for (int i = 0; i < num_pt; i++) {
            PointData& pt_data = pt_meas[i];
            pt_data.if_valid = false;
            if (!(spline->numKnots() == 4) && !(pt_data.time_ns <= spline->maxTimeNs() && pt_data.time_ns >= spline->maxTimeNs() - 4*spline->getKnotTimeIntervalNs())) {                           
                continue;
            }
            Association::pointBodyToWorld(pt_data.time_ns, spline, pt_data.pt, pt_data.pt_w, pt_data.t_bl, pt_data.q_bl);
            std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
            pt_data.nearest_points.clear();
            ikdtree->Nearest_Search(pt_data.pt_w, NUM_MATCH_POINTS, pt_data.nearest_points, pointSearchSqDis, 2.236); 
            if (pt_data.nearest_points.size() >= (size_t)NUM_MATCH_POINTS && pointSearchSqDis[NUM_MATCH_POINTS - 1] < 5) {     
                Eigen::Vector4f pabcd;       
                pabcd.setZero();
                if (CommonUtils::esti_plane(pabcd, pt_data.nearest_points, 0.1f)) {
                    float pd2 = pabcd(0) * pt_data.pt_w.x + pabcd(1) * pt_data.pt_w.y + pabcd(2) * pt_data.pt_w.z + pabcd(3);
                    if (pt_data.pt_b.norm() > 81.0 * pd2 * pd2) {
                        pt_data.if_valid = true;
                        pt_data.normvec = Eigen::Vector3d(pabcd[0], pabcd[1], pabcd[2]);
                        pt_data.dist = pabcd(3);                            
                    }
                }
            }
        }
        for (int i = 0; i < num_pt; i++) {
            if (pt_meas[i].if_valid) {
                effect_num_k++;
            } 
        }    
    }        
};