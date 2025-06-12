#ifndef OV_CORE_TRACK_KLT_H
#define OV_CORE_TRACK_KLT_H

#include "TrackBase.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>   // GPU LK 加速
#include <opencv2/core/cuda.hpp>

#include "pzj/pyramid_gpu.h"          // 第 2 步创建的头文件

namespace ov_core {

class TrackKLT : public TrackBase {
public:
  explicit TrackKLT(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras,
                    int numfeats, int numaruco, bool stereo,
                    HistogramMethod histmethod,
                    int fast_threshold, int gridx, int gridy, int minpxdist)
      : TrackBase(cameras, numfeats, numaruco, stereo, histmethod),
        threshold(fast_threshold), grid_x(gridx), grid_y(gridy),
        min_px_dist(minpxdist) {
    // 初始化 GPU 稀疏金字塔 LK
    gpu_lk_ = cv::cuda::SparsePyrLKOpticalFlow::create(win_size, pyr_levels);
  }

  void feed_new_camera(const CameraData &message) override;

protected:
  void feed_monocular(const CameraData &message, size_t msg_id);
  void feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right);
  void perform_detection_monocular(const std::vector<cv::Mat> &img0pyr,
                                   const cv::Mat &mask0,
                                   std::vector<cv::KeyPoint> &pts0,
                                   std::vector<size_t> &ids0);
  void perform_detection_stereo(const std::vector<cv::Mat> &img0pyr,
                                const std::vector<cv::Mat> &img1pyr,
                                const cv::Mat &mask0, const cv::Mat &mask1,
                                size_t cam_id_left, size_t cam_id_right,
                                std::vector<cv::KeyPoint> &pts0,
                                std::vector<cv::KeyPoint> &pts1,
                                std::vector<size_t> &ids0,
                                std::vector<size_t> &ids1);
  void perform_matching(const std::vector<cv::Mat> &img0pyr,
                        const std::vector<cv::Mat> &img1pyr,
                        std::vector<cv::KeyPoint> &pts0,
                        std::vector<cv::KeyPoint> &pts1,
                        size_t id0, size_t id1,
                        std::vector<uchar> &mask_out);
  void perform_matching_gpu(const std::vector<cv::cuda::GpuMat> &img0pyr,
                            const std::vector<cv::cuda::GpuMat> &img1pyr,
                            std::vector<cv::KeyPoint> &pts0,
                            std::vector<cv::KeyPoint> &pts1,
                            size_t id0, size_t id1,
                            std::vector<uchar> &mask_out);

  int threshold;
  int grid_x, grid_y;
  int min_px_dist;
  int pyr_levels = 5;
  cv::Size win_size = cv::Size(15, 15);

  std::map<size_t, std::vector<cv::Mat>> img_pyramid_last;
  std::map<size_t, cv::Mat> img_curr;
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_curr;
  std::unordered_map<size_t, std::vector<cv::cuda::GpuMat>> d_img_pyramid_curr_;
  std::unordered_map<size_t, std::vector<cv::cuda::GpuMat>> d_img_pyramid_last_;
  cv::cuda::Stream cv_stream_;

  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> gpu_lk_;   // GPU LK
};

} // namespace ov_core

#endif /* OV_CORE_TRACK_KLT_H */
