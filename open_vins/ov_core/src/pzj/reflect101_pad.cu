#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp> // ⭐ 访问底层 cudaStream_t

/* ----------------- GPU Kernel ----------------- */
template <typename T>
__global__ void reflect101Kernel(const T *src, int srcW, int srcH, size_t srcStep, T *dst, size_t dstStep, int padX, int padY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; // 0 … dstW-1
  int y = blockIdx.y * blockDim.y + threadIdx.y; // 0 … dstH-1
  int dstW = srcW + 2 * padX;
  int dstH = srcH + 2 * padY;
  if (x >= dstW || y >= dstH)
    return;

  // --------- 计算镜像坐标 (Reflect-101) ----------
  int xm = (x < padX) ? (padX * 2 - x - 1) : (x >= padX + srcW) ? (2 * (padX + srcW) - x - 1) : (x - padX);

  int ym = (y < padY) ? (padY * 2 - y - 1) : (y >= padY + srcH) ? (2 * (padY + srcH) - y - 1) : (y - padY);

  const T *src_line = (const T *)((const unsigned char *)src + ym * srcStep);
  T *dst_line = (T *)((unsigned char *)dst + y * dstStep);
  dst_line[x] = src_line[xm];
}

/* ---------------- Host 封装函数 ---------------- */
namespace ov_core {

cv::cuda::GpuMat reflect101Pad(const cv::cuda::GpuMat &src, int padY, int padX, cv::cuda::Stream &stream) {
  int dstW = src.cols + 2 * padX;
  int dstH = src.rows + 2 * padY;

  cv::cuda::GpuMat dst(dstH, dstW, src.type());

  size_t strideSrc = src.step; // 按像素
  size_t strideDst = dst.step;

  dim3 block(32, 8);
  dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);

  /* ⭐ 关键：把 OpenCV Stream 转成 cudaStream_t */
  cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

  /* ---------- Kernel Launch 按通道分派 ---------- */
  if (src.channels() == 1) {
    reflect101Kernel<uchar>
        <<<grid, block, 0, cuda_stream>>>(src.ptr<uchar>(), src.cols, src.rows, strideSrc, dst.ptr<uchar>(), strideDst, padX, padY);
  } else if (src.channels() == 3) {
    reflect101Kernel<uchar3>
        <<<grid, block, 0, cuda_stream>>>(src.ptr<uchar3>(), src.cols, src.rows, strideSrc, dst.ptr<uchar3>(), strideDst, padX, padY);
  } else if (src.channels() == 4) {
    reflect101Kernel<uchar4>
        <<<grid, block, 0, cuda_stream>>>(src.ptr<uchar4>(), src.cols, src.rows, strideSrc, dst.ptr<uchar4>(), strideDst, padX, padY);
  } else {
    CV_Error(cv::Error::StsUnsupportedFormat, "Reflect101: unsupported channel count");
  }

  return dst; // GpuMat 持有显存，供后续使用
}

} // namespace ov_core
