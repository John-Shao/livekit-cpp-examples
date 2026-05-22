/*
 * Single-process bidirectional smoke test for ATK-DLRV1126B board.
 *
 * - Publishes a synthetic RGBA gradient video track ("board-cam") and a data
 *   track ("board-data") so the remote peer (e.g. mobile LiveKit test app)
 *   can verify board → peer media flow.
 * - Implements RoomDelegate::onTrackSubscribed to register frame callbacks
 *   for EVERY remote track, regardless of identity or track name. Logs a
 *   per-stream counter every 5s so you can verify peer → board media flow.
 * - Auto-exits after 30 seconds — matches plan.md Phase 5 Go standard.
 *
 * Usage:
 *   BoardLoopback <ws-url> <token>
 *   LIVEKIT_URL=wss://... LIVEKIT_TOKEN=eyJ... BoardLoopback
 *
 * Token must grant `roomJoin` for the target room. The board's identity is
 * whatever the token's `sub` claim says.
 */

// Phase 8.4.4: HTTP API server (cpp-httplib header-only)。我们的 JSON 用例
// 仅 4 个：解析 join body 取 url+token、生成 status 响应、生成 join
// response、错误响应。手写 30 行 minimal helper 避免 nlohmann_json 依赖
// （Buildroot 不带，FetchContent 又卡 GitHub），BoardLoopback 二进制更瘦。
#include <httplib.h>

#include "livekit/audio_frame.h"
#include "livekit/audio_processing_module.h"
#include "livekit/audio_source.h"
#include "livekit/livekit.h"
#include "livekit/local_audio_track.h"
#include "livekit/local_data_track.h"
#include "livekit/local_participant.h"
#include "livekit/local_video_track.h"
#include "livekit/participant.h"
#include "livekit/room.h"
#include "livekit/room_delegate.h"
#include "livekit/room_event_types.h"
#include "livekit/track.h"
#include "livekit/video_frame.h"
#include "livekit/video_source.h"

#include <cctype>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define BOARD_LOOPBACK_HAVE_NEON 1
#endif

#include <alsa/asoundlib.h>

#include <execinfo.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <linux/videodev2.h>

#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm/drm_fourcc.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace livekit;

// Video resolution presets — selected via BOARD_LOOPBACK_VIDEO_RES env
// (sd / hd / fhd, default hd). Each triplet drives V4L2 capture, framerate
// hint via S_PARM, and the publish-side max_bitrate so simulcast / BWE
// don't starve high-resolution layers (default WebRTC bitrate allocator
// gives 720p only ~30 kbps without an explicit cap, producing block-heavy
// garbage on the wire).
struct VideoResolution {
  int width;
  int height;
  int fps;
  std::uint64_t max_bitrate_bps; // sane WebRTC defaults per resolution tier
  const char *name;
};
// bits-per-pixel-per-frame at these caps:
//   SD  640×360  @ 30fps × 1.5 Mbps = 0.217 bpp   (16:9, 同 HD/FHD 一致;
//                                                  上手机 1080+ 屏放大有点糊)
//   HD  1280×720 @ 30fps × 2.5 Mbps = 0.090 bpp   (实测最佳)
//   FHD 1920×1080@ 25fps × 5.5 Mbps = 0.106 bpp   (略高于 HD 密度，补偿 1080p 细节)
// H.264 Constrained Baseline 没有 CABAC / B-frames / 8x8 transform，
// 比 Main 大约要多 20% 码率才能拿到同样画质。
constexpr VideoResolution kResSD  = { 640,  360, 30, 1'500'000, "SD"  };
constexpr VideoResolution kResHD  = {1280,  720, 30, 2'500'000, "HD"  };
constexpr VideoResolution kResFHD = {1920, 1080, 25, 5'500'000, "FHD" };

constexpr VideoResolution kDefaultRes = kResHD;
constexpr const char *kVideoTrackName = "board-cam";
constexpr const char *kAudioTrackName = "board-mic";
constexpr int kAudioSampleRate = 48000;
constexpr int kAudioChannels = 1;
constexpr int kAudioFrameMs = 10; // SDK direct-capture 要求 10ms 帧 (= 480 @48k)
constexpr int kAudioSamplesPerChannel = kAudioSampleRate * kAudioFrameMs / 1000;
constexpr double kAudioToneHz = 440.0;     // A4 note，可听见的低频
constexpr double kAudioAmplitude = 6000.0; // ~ -15 dBFS, 不刺耳
constexpr int kReportEverySeconds = 5;

static std::atomic<bool> g_running{true};
// handleSignal 的实际定义在文件后面，等 g_meeting_state 完整声明之后。
// 这里仅前向声明，让 std::signal 注册时找得到符号。
static void handleSignal(int);

static std::string getenvOrEmpty(const char *name) {
  const char *v = std::getenv(name);
  return v ? std::string(v) : std::string{};
}

// Per-(identity, track) counters
struct LoopbackStats {
  std::atomic<std::uint64_t> video_frames{0};
  std::atomic<std::uint64_t> audio_frames{0};
};

struct StreamKey {
  std::string identity;
  std::string track;
  bool operator<(const StreamKey &o) const {
    if (identity != o.identity)
      return identity < o.identity;
    return track < o.track;
  }
};

class StatsRegistry {
public:
  std::shared_ptr<LoopbackStats> get(const std::string &id,
                                   const std::string &tr) {
    std::lock_guard<std::mutex> lk(mu_);
    auto &s = map_[StreamKey{id, tr}];
    if (!s)
      s = std::make_shared<LoopbackStats>();
    return s;
  }
  void print(std::ostream &os) {
    std::lock_guard<std::mutex> lk(mu_);
    if (map_.empty()) {
      os << "  (no remote streams yet)\n";
      return;
    }
    for (const auto &[k, s] : map_) {
      os << "  " << k.identity << ":" << k.track
         << " video=" << s->video_frames.load()
         << " audio=" << s->audio_frames.load() << "\n";
    }
  }

private:
  std::mutex mu_;
  std::map<StreamKey, std::shared_ptr<LoopbackStats>> map_;
};

static StatsRegistry g_stats;

// Phase 7.4: explicit, frame-level Acoustic Echo Cancellation via the SDK's
// standalone livekit::AudioProcessingModule (AEC3 + NS + HPF). The wrapper
// runs *outside* the PCF's internal APM — we feed it the playback frame via
// processReverseStream() before snd_pcm_writei, and the captured mic frame
// via processStream() before audio_source->captureFrame(). With both ends
// fed and a delay hint set via setStreamDelayMs, AEC3 can subtract the
// speaker→mic acoustic loop.
//
// Two earlier attempts (Phase 7.3 ADM-driven playback, Phase 7.4 ADM-driven
// capture+playback) both wired the SDK's internal APM but didn't deliver
// perceptible echo cancellation — the LocalAudioTrack ends up taking audio
// from AudioTrackSource's sink chain regardless, which doesn't pass through
// AEC even when ADM does. Bypassing PCF's APM and using the explicit module
// here puts AEC right where the data flows.
static std::unique_ptr<AudioProcessingModule> g_apm;
static int g_apm_delay_ms = 0;
static std::atomic<bool> g_apm_ready{false};

// ---------------------------------------------------------------
// ALSA player: lazy-init PCM playback to plughw:0,0 (ES8389 codec on
// ATK-DLRV1126B). Producer-consumer split — SDK audio callback only
// enqueues into the FIFO and returns immediately; a dedicated writer
// thread drains the FIFO into snd_pcm_writei. This decouples the
// SDK's subscription thread from the codec's hardware pacing and
// avoids back-pressure-induced underruns.
// ---------------------------------------------------------------
// AudioMixer — Phase 8.4.3
// ---------------------------------------------------------------
// 多 peer 音频软混音器。每 peer 一个 100ms ring buffer，写者是各 peer 的
// audio callback（独立线程），读者是 ALSA writer 线程（pull 模型）。
//
// 关键设计：**pull 而不是 push**。原 AlsaPlayer 是 push 模型，每个 peer
// callback 各自 enqueue ALSA 队列 → N 个 peer 的 sample rate 叠加进队列，
// drain 不过来就丢。新 pull 模型：ALSA writer 线程按 10ms 节奏 pull 一帧
// 出来，混音器把各 peer ring 当前的 480 samples 求和 + clip 输出，单源
// 馈给 ALSA + AEC reverse → 帧级语义干净。
//
// SPSC 性质：每个 peer 一个 ring，仅一个 SDK 线程写、ALSA 线程读。整体
// 用单 mutex 保护 ring map + ring 状态，临界区很短（拷贝 samples），
// 10 peer × 100Hz × 480 samples 的负载可忽略。
class AudioMixer {
public:
  // 500ms @ 48kHz mono = 24000 samples × 2 byte = 48 KB / peer。
  // 大于 ALSA period 颗粒（即便配 1s buffer，period 可能 250ms），保证
  // writer 短暂阻塞期间 push 不溢出。10 peer × 48 KB = 480 KB，可接受。
  static constexpr std::size_t kRingCapacitySamples = 24000;
  static constexpr int kSampleRate = 48000;
  static constexpr int kChannels = 1;

  struct Ring {
    std::array<std::int16_t, kRingCapacitySamples> buf{};
    std::size_t write_pos = 0;
    std::size_t read_pos = 0;
    std::size_t fill = 0; // 已写但未读的样本数
  };

  // 写入指定 peer 的 ring。ring 满时丢最旧 sample（保新弃旧 = 低延迟）。
  void write(const std::string &id, const std::int16_t *samples,
             std::size_t n) {
    if (n == 0)
      return;
    std::lock_guard<std::mutex> lk(mu_);
    auto &ring = rings_[id];
    for (std::size_t i = 0; i < n; ++i) {
      if (ring.fill >= kRingCapacitySamples) {
        // overrun：丢最旧 1 sample
        ring.read_pos = (ring.read_pos + 1) % kRingCapacitySamples;
        ring.fill -= 1;
        dropped_.fetch_add(1, std::memory_order_relaxed);
      }
      ring.buf[ring.write_pos] = samples[i];
      ring.write_pos = (ring.write_pos + 1) % kRingCapacitySamples;
      ring.fill += 1;
    }
  }

  // 从所有 peer ring 各取 n samples，求和 + clip 写到 out。
  // 返回是否有任何 peer 提供数据（false = 静音输出）。
  bool pull(std::int16_t *out, std::size_t n) {
    // 用 int32 累加器避免溢出
    std::vector<std::int32_t> acc(n, 0);
    bool any = false;
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (auto &kv : rings_) {
        auto &ring = kv.second;
        if (ring.fill == 0)
          continue;
        any = true;
        const std::size_t take = std::min(ring.fill, n);
        for (std::size_t i = 0; i < take; ++i) {
          acc[i] += ring.buf[ring.read_pos];
          ring.read_pos = (ring.read_pos + 1) % kRingCapacitySamples;
        }
        ring.fill -= take;
        // take < n 时尾部留 0，别的 peer 可能补上去（或全 0 = 静音）
      }
    }
    for (std::size_t i = 0; i < n; ++i) {
      std::int32_t v = acc[i];
      if (v > std::numeric_limits<std::int16_t>::max())
        v = std::numeric_limits<std::int16_t>::max();
      else if (v < std::numeric_limits<std::int16_t>::min())
        v = std::numeric_limits<std::int16_t>::min();
      out[i] = static_cast<std::int16_t>(v);
    }
    return any;
  }

  void removePeer(const std::string &id) {
    std::lock_guard<std::mutex> lk(mu_);
    rings_.erase(id);
  }

  std::uint64_t dropped() const {
    return dropped_.load(std::memory_order_relaxed);
  }

private:
  std::mutex mu_;
  std::unordered_map<std::string, Ring> rings_;
  std::atomic<std::uint64_t> dropped_{0};
};

static AudioMixer g_mixer;

class AlsaPlayer {
public:
  // Init from first audio frame's params and start the writer thread.
  bool ensureOpen(int sample_rate, int num_channels) {
    std::lock_guard<std::mutex> lk(open_mu_);
    if (pcm_) {
      if (sample_rate != sr_ || num_channels != ch_) {
        std::cerr << "[loopback] alsa: ignoring mismatched frame format\n";
        return false;
      }
      return true;
    }
    // 板上跑着 PulseAudio (PID 658) 占用 plughw 直通会跟它抢硬件，质量差。
    // 走 ALSA "default" 设备，PulseAudio 接管混音/重采样，质量更稳。
    // 允许通过 ALSA_PCM_DEVICE 环境变量覆盖（比如想 bypass pulse 时设
    // ALSA_PCM_DEVICE=plughw:0,0 直通）。
    const char *dev_env = std::getenv("ALSA_PCM_DEVICE");
    const char *device = (dev_env && *dev_env) ? dev_env : "default";
    snd_pcm_t *pcm = nullptr;
    int err = snd_pcm_open(&pcm, device, SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0) {
      std::cerr << "[loopback] alsa snd_pcm_open(" << device << ") failed: "
                << snd_strerror(err) << "\n";
      return false;
    }
    // Phase 8.4.3: 把 latency 从 1s 缩到 200ms，让 writer thread 的
    // snd_pcm_writei 阻塞颗粒变小（从 ~250ms period 跌到 ~50ms 量级），
    // pull 不至于长时间停顿 → AudioMixer ring 溢出概率降到底。
    // 200ms 仍留足 jitter buffer 余量给手机/Web 端 RTP 抖动。
    err = snd_pcm_set_params(pcm, SND_PCM_FORMAT_S16_LE,
                             SND_PCM_ACCESS_RW_INTERLEAVED, num_channels,
                             sample_rate, /*soft_resample=*/1,
                             /*latency_us=*/200000); // 200ms 缓冲
    if (err < 0) {
      std::cerr << "[loopback] alsa snd_pcm_set_params failed: "
                << snd_strerror(err) << "\n";
      snd_pcm_close(pcm);
      return false;
    }
    pcm_ = pcm;
    sr_ = sample_rate;
    ch_ = num_channels;
    std::cout << "[loopback] alsa playback opened: " << device << " "
              << sample_rate << "Hz " << num_channels << "ch s16le 200ms-buf\n";
    writer_ = std::thread(&AlsaPlayer::writerLoop, this);
    return true;
  }

  std::uint64_t underruns() const {
    return underruns_.load(std::memory_order_relaxed);
  }

  ~AlsaPlayer() {
    stop_.store(true, std::memory_order_release);
    if (writer_.joinable())
      writer_.join();
    if (pcm_) {
      snd_pcm_drain(pcm_);
      snd_pcm_close(pcm_);
      pcm_ = nullptr;
    }
  }

private:
  // Phase 8.4.3: pull 模型。每 10ms 从 g_mixer 拉一帧 480 samples，
  // 喂 APM AEC reverse stream（让 AEC 看到混好的总声），再 snd_pcm_writei。
  // snd_pcm_writei 阻塞到 ALSA 接受 → 自然按 48kHz/480 = 10ms 节拍。
  // mixer 没数据时输出静音（不能不写，否则 ALSA underrun 出 pop 声）。
  void writerLoop() {
    constexpr std::size_t kSamplesPerTick = 480; // 10ms @ 48kHz mono
    std::vector<std::int16_t> mixed(kSamplesPerTick);
    while (!stop_.load(std::memory_order_acquire)) {
      if (!pcm_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      // 拉一帧混音；mixer 空时返回静音（all-zero buf 已由 pull 写好）
      g_mixer.pull(mixed.data(), kSamplesPerTick);

      // 喂 AEC reverse —— processReverseStream 移动 frame 内部 vector，
      // 所以传 copy；播放仍用 mixed 原 buf。
      if (g_apm_ready.load(std::memory_order_acquire)) {
        AudioFrame reverse_frame(
            std::vector<std::int16_t>(mixed.begin(), mixed.end()),
            AudioMixer::kSampleRate, AudioMixer::kChannels,
            static_cast<int>(kSamplesPerTick));
        try {
          g_apm->processReverseStream(reverse_frame);
        } catch (const std::exception &e) {
          std::cerr << "[loopback] APM processReverseStream err: " << e.what()
                    << "\n";
        }
      }

      // 阻塞 write，ALSA 自然按硬件节拍 pace 这个循环
      snd_pcm_sframes_t written =
          snd_pcm_writei(pcm_, mixed.data(), kSamplesPerTick);
      if (written < 0) {
        const int err = static_cast<int>(written);
        if (err == -EPIPE)
          underruns_.fetch_add(1, std::memory_order_relaxed);
        snd_pcm_recover(pcm_, err, /*silent=*/1);
      }
    }
  }

  std::mutex open_mu_;
  snd_pcm_t *pcm_ = nullptr;
  int sr_ = 0;
  int ch_ = 0;

  std::atomic<bool> stop_{false};
  std::thread writer_;

  std::atomic<std::uint64_t> underruns_{0};
};

static AlsaPlayer g_alsa;

// ---------------------------------------------------------------
// V4L2 multi-planar capture for /dev/video-camera0 (rkisp_mainpath
// on RV1126B). Negotiates NV12 at the requested resolution, mmaps
// 4 buffers, streams in blocking-poll mode. dequeue() returns the
// next NV12 frame as a freshly allocated vector ready to hand to
// VideoFrame.
// ---------------------------------------------------------------
class V4l2Capture {
public:
  bool open(const char *device, int width, int height, int fps = 30) {
    fd_ = ::open(device, O_RDWR | O_CLOEXEC);
    if (fd_ < 0) {
      std::cerr << "[loopback] v4l2 open(" << device
                << ") failed: " << std::strerror(errno) << "\n";
      return false;
    }
    width_ = width;
    height_ = height;
    fps_ = fps;

    v4l2_capability cap{};
    if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
      std::cerr << "[loopback] v4l2 QUERYCAP failed: "
                << std::strerror(errno) << "\n";
      return false;
    }
    if (!(cap.capabilities &
          (V4L2_CAP_VIDEO_CAPTURE_MPLANE | V4L2_CAP_DEVICE_CAPS))) {
      std::cerr << "[loopback] v4l2 device not multi-planar capture\n";
      return false;
    }

    // Set format: width × height NV12, single plane (NV12 is conceptually
    // 1 plane of (Y + interleaved UV) bytes).
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = width;
    fmt.fmt.pix_mp.height = height;
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
    fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
    fmt.fmt.pix_mp.num_planes = 1;
    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
      std::cerr << "[loopback] v4l2 S_FMT failed: " << std::strerror(errno)
                << "\n";
      return false;
    }
    width_ = fmt.fmt.pix_mp.width;
    height_ = fmt.fmt.pix_mp.height;
    plane_size_ = fmt.fmt.pix_mp.plane_fmt[0].sizeimage;

    // Negotiate framerate via S_PARM. The driver may quantize to whatever
    // the sensor + rkisp pipeline supports — we read back actual fps to
    // log the truth. Older drivers return EINVAL on S_PARM for capture;
    // treat as soft warning, fall back to whatever default the driver picks.
    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = static_cast<__u32>(fps);
    if (ioctl(fd_, VIDIOC_S_PARM, &parm) < 0) {
      std::cerr << "[loopback] v4l2 S_PARM(fps=" << fps << ") not supported: "
                << std::strerror(errno) << " (using driver default)\n";
    } else if (parm.parm.capture.timeperframe.denominator) {
      fps_ = static_cast<int>(parm.parm.capture.timeperframe.denominator /
                              std::max<__u32>(parm.parm.capture.timeperframe.numerator, 1));
    }

    std::cout << "[loopback] v4l2 negotiated: " << width_ << "x" << height_
              << "@" << fps_ << "fps NV12 plane_size=" << plane_size_ << "\n";
    return true;
  }

  int fps() const { return fps_; }

  bool startStream(int buf_count = 4) {
    v4l2_requestbuffers reqbuf{};
    reqbuf.count = buf_count;
    reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    reqbuf.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd_, VIDIOC_REQBUFS, &reqbuf) < 0) {
      std::cerr << "[loopback] v4l2 REQBUFS failed: "
                << std::strerror(errno) << "\n";
      return false;
    }

    buffers_.resize(reqbuf.count);
    for (std::uint32_t i = 0; i < reqbuf.count; ++i) {
      v4l2_buffer buf{};
      v4l2_plane planes[1]{};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;
      buf.length = 1;
      buf.m.planes = planes;
      if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
        std::cerr << "[loopback] v4l2 QUERYBUF " << i
                  << " failed: " << std::strerror(errno) << "\n";
        return false;
      }
      buffers_[i].length = planes[0].length;
      buffers_[i].start =
          mmap(nullptr, planes[0].length, PROT_READ | PROT_WRITE, MAP_SHARED,
               fd_, planes[0].m.mem_offset);
      if (buffers_[i].start == MAP_FAILED) {
        std::cerr << "[loopback] v4l2 mmap " << i
                  << " failed: " << std::strerror(errno) << "\n";
        return false;
      }
      // Queue this buffer for the driver
      v4l2_buffer qbuf{};
      v4l2_plane qplanes[1]{};
      qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
      qbuf.memory = V4L2_MEMORY_MMAP;
      qbuf.index = i;
      qbuf.length = 1;
      qbuf.m.planes = qplanes;
      if (ioctl(fd_, VIDIOC_QBUF, &qbuf) < 0) {
        std::cerr << "[loopback] v4l2 initial QBUF " << i
                  << " failed: " << std::strerror(errno) << "\n";
        return false;
      }
    }

    int t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(fd_, VIDIOC_STREAMON, &t) < 0) {
      std::cerr << "[loopback] v4l2 STREAMON failed: "
                << std::strerror(errno) << "\n";
      return false;
    }
    std::cout << "[loopback] v4l2 stream on, " << buffers_.size()
              << " buffers\n";
    return true;
  }

  // Returns true on success; out_data populated with NV12 bytes
  // (Y plane + UV plane). Returns false on timeout or error.
  bool dequeue(std::vector<std::uint8_t> &out_data, int timeout_ms = 1000) {
    pollfd pfd{fd_, POLLIN, 0};
    int pr = poll(&pfd, 1, timeout_ms);
    if (pr <= 0)
      return false;

    v4l2_buffer buf{};
    v4l2_plane planes[1]{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.length = 1;
    buf.m.planes = planes;
    if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
      if (errno == EAGAIN)
        return false;
      std::cerr << "[loopback] v4l2 DQBUF failed: "
                << std::strerror(errno) << "\n";
      return false;
    }

    const std::size_t bytes_used =
        std::min<std::size_t>(planes[0].bytesused, buffers_[buf.index].length);
    out_data.assign(
        static_cast<std::uint8_t *>(buffers_[buf.index].start),
        static_cast<std::uint8_t *>(buffers_[buf.index].start) + bytes_used);

    // Re-queue the buffer
    v4l2_buffer qbuf{};
    v4l2_plane qplanes[1]{};
    qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    qbuf.memory = V4L2_MEMORY_MMAP;
    qbuf.index = buf.index;
    qbuf.length = 1;
    qbuf.m.planes = qplanes;
    if (ioctl(fd_, VIDIOC_QBUF, &qbuf) < 0) {
      std::cerr << "[loopback] v4l2 re-QBUF failed: "
                << std::strerror(errno) << "\n";
    }
    return true;
  }

  int width() const { return width_; }
  int height() const { return height_; }

  void close() {
    if (fd_ >= 0) {
      int t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
      ioctl(fd_, VIDIOC_STREAMOFF, &t);
      for (auto &b : buffers_) {
        if (b.start && b.start != MAP_FAILED)
          munmap(b.start, b.length);
      }
      buffers_.clear();
      ::close(fd_);
      fd_ = -1;
    }
  }

  ~V4l2Capture() { close(); }

private:
  struct Buf {
    void *start = nullptr;
    std::size_t length = 0;
  };
  int fd_ = -1;
  int width_ = 0;
  int height_ = 0;
  int fps_ = 30;
  std::uint32_t plane_size_ = 0;
  std::vector<Buf> buffers_;
};

// ---------------------------------------------------------------
// DrmDisplay: render incoming NV12 frames directly to the board's
// MIPI panel via DRM/KMS plane composition. No EGL/GL — RV1126B has
// no Mali GPU. Dumb buffers are CPU-mapped, we memcpy NV12 in, then
// drmModeSetPlane.
//
// Caller must ensure no other DRM master holds the card (e.g. stop
// weston: `pkill weston`). drmSetMaster will be attempted; failure
// likely means another compositor is holding the card.
// ---------------------------------------------------------------
class DrmDisplay {
public:
  bool open() {
    fd_ = ::open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (fd_ < 0) {
      std::cerr << "[drm] open(/dev/dri/card0) failed: "
                << std::strerror(errno) << "\n";
      return false;
    }
    if (drmSetMaster(fd_) < 0) {
      std::cerr << "[drm] drmSetMaster failed: " << std::strerror(errno)
                << " — another compositor (weston?) is holding DRM master\n";
      ::close(fd_);
      fd_ = -1;
      return false;
    }
    drmModeRes *res = drmModeGetResources(fd_);
    if (!res) {
      std::cerr << "[drm] drmModeGetResources failed\n";
      return false;
    }

    // Find first connected connector
    drmModeConnector *conn = nullptr;
    for (int i = 0; i < res->count_connectors; ++i) {
      drmModeConnector *c = drmModeGetConnector(fd_, res->connectors[i]);
      if (c && c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) {
        conn = c;
        break;
      }
      if (c)
        drmModeFreeConnector(c);
    }
    if (!conn) {
      std::cerr << "[drm] no connected connector\n";
      drmModeFreeResources(res);
      return false;
    }
    conn_id_ = conn->connector_id;
    mode_ = conn->modes[0];
    screen_w_ = mode_.hdisplay;
    screen_h_ = mode_.vdisplay;
    std::cout << "[drm] connector " << conn_id_ << " " << screen_w_ << "x"
              << screen_h_ << "@" << mode_.vrefresh << "Hz\n";

    // Pick CRTC via current encoder
    drmModeEncoder *enc = drmModeGetEncoder(fd_, conn->encoder_id);
    if (enc) {
      crtc_id_ = enc->crtc_id;
      drmModeFreeEncoder(enc);
    } else {
      // Fallback: pick first possible CRTC
      for (int i = 0; i < res->count_encoders; ++i) {
        drmModeEncoder *e = drmModeGetEncoder(fd_, res->encoders[i]);
        if (!e) continue;
        if (e->possible_crtcs && res->count_crtcs > 0) {
          crtc_id_ = res->crtcs[0];
          drmModeFreeEncoder(e);
          break;
        }
        drmModeFreeEncoder(e);
      }
    }
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);

    if (!crtc_id_) {
      std::cerr << "[drm] no CRTC\n";
      return false;
    }

    // Find an NV12-capable plane attached to our CRTC
    drmSetClientCap(fd_, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
    drmModePlaneRes *pres = drmModeGetPlaneResources(fd_);
    if (!pres) {
      std::cerr << "[drm] drmModeGetPlaneResources failed\n";
      return false;
    }
    for (std::uint32_t i = 0; i < pres->count_planes; ++i) {
      drmModePlane *p = drmModeGetPlane(fd_, pres->planes[i]);
      if (!p) continue;
      bool can_use = false;
      // Plane must be usable on our CRTC and support NV12
      const int crtc_idx = crtcIndex();
      if (p->possible_crtcs & (1u << crtc_idx)) {
        for (std::uint32_t f = 0; f < p->count_formats; ++f) {
          if (p->formats[f] == DRM_FORMAT_NV12) {
            can_use = true;
            break;
          }
        }
      }
      if (can_use) {
        plane_id_ = p->plane_id;
        std::cout << "[drm] using plane " << plane_id_ << " (NV12)\n";
        drmModeFreePlane(p);
        break;
      }
      drmModeFreePlane(p);
    }
    drmModeFreePlaneResources(pres);
    if (!plane_id_) {
      std::cerr << "[drm] no NV12-capable plane found\n";
      return false;
    }

    return true;
  }

  // Allocate a pair of dumb buffers sized for the source video frame.
  // Source frame size is dynamic (we use the first frame's dimensions).
  bool ensureBuffers(int w, int h) {
    if (buf_w_ == w && buf_h_ == h)
      return true;
    freeBuffers();
    buf_w_ = w;
    buf_h_ = h;
    for (int i = 0; i < 2; ++i) {
      DumbBuf &b = bufs_[i];
      drm_mode_create_dumb cd{};
      cd.width = w;
      cd.height = h * 3 / 2; // NV12 = 1.5 * w * h
      cd.bpp = 8;
      if (drmIoctl(fd_, DRM_IOCTL_MODE_CREATE_DUMB, &cd) < 0) {
        std::cerr << "[drm] CREATE_DUMB failed: " << std::strerror(errno)
                  << "\n";
        return false;
      }
      b.handle = cd.handle;
      b.size = cd.size;
      b.pitch = cd.pitch;

      drm_mode_map_dumb md{};
      md.handle = b.handle;
      if (drmIoctl(fd_, DRM_IOCTL_MODE_MAP_DUMB, &md) < 0) {
        std::cerr << "[drm] MAP_DUMB failed: " << std::strerror(errno) << "\n";
        return false;
      }
      b.mapped = static_cast<std::uint8_t *>(
          mmap(nullptr, b.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_,
               md.offset));
      if (b.mapped == MAP_FAILED) {
        std::cerr << "[drm] mmap dumb failed: " << std::strerror(errno)
                  << "\n";
        return false;
      }
      std::memset(b.mapped, 0, b.size);

      // Build FB with NV12: plane 0 = Y at offset 0 with pitch.
      // plane 1 = UV right after Y, pitch same as Y (interleaved 8-bit pairs).
      std::uint32_t handles[4] = {b.handle, b.handle, 0, 0};
      std::uint32_t pitches[4] = {b.pitch, b.pitch, 0, 0};
      std::uint32_t offsets[4] = {0, b.pitch * static_cast<std::uint32_t>(h),
                                  0, 0};
      if (drmModeAddFB2(fd_, w, h, DRM_FORMAT_NV12, handles, pitches,
                        offsets, &b.fb_id, 0) < 0) {
        std::cerr << "[drm] AddFB2 failed: " << std::strerror(errno) << "\n";
        return false;
      }
    }
    std::cout << "[drm] dumb buffers ready: 2x " << w << "x" << h
              << " NV12\n";
    return true;
  }

  void renderNV12(const std::uint8_t *src, int src_w, int src_h) {
    // Simulcast/adaptive bitrate 的最低层有时会塌到 8x8 之类极小尺寸，
    // 拉伸到 720x1280 倍率超过 driver 的合法上限（drmModeSetPlane 报
    // EOVERFLOW），后续 frame 也连环失败。设个最小阈值丢弃这类帧，
    // 让屏幕保留上一可用帧。
    if (src_w < 64 || src_h < 64) {
      return;
    }
    // Phase 8.4.1: 多 peer 场景下，多个 video track callback 并发调
    // renderNV12 → ensureBuffers 检测到 size 变化时 free/realloc dumb
    // buffer，另一路 callback 拿着已 free 的 mapped 指针写入触发 segv。
    // 实践中 99% 的 callback 在 VideoRouter::renderIfActive 入口就被
    // 早 return 不会进这里，但即便如此也要兜底加锁。
    std::lock_guard<std::mutex> render_lock(render_mutex_);
    if (!ensureBuffers(src_w, src_h))
      return;
    DumbBuf &b = bufs_[next_];
    next_ ^= 1;

    // Copy Y plane (src_w x src_h)
    if (b.pitch == static_cast<std::uint32_t>(src_w)) {
      std::memcpy(b.mapped, src, static_cast<std::size_t>(src_w) * src_h);
    } else {
      for (int y = 0; y < src_h; ++y) {
        std::memcpy(b.mapped + y * b.pitch, src + y * src_w, src_w);
      }
    }
    // Copy UV plane (src_w x src_h/2 → right after Y in the dumb buffer)
    const std::size_t uv_offset_dst = b.pitch * static_cast<std::size_t>(src_h);
    const std::uint8_t *uv_src = src + static_cast<std::size_t>(src_w) * src_h;
    if (b.pitch == static_cast<std::uint32_t>(src_w)) {
      std::memcpy(b.mapped + uv_offset_dst, uv_src,
                  static_cast<std::size_t>(src_w) * src_h / 2);
    } else {
      for (int y = 0; y < src_h / 2; ++y) {
        std::memcpy(b.mapped + uv_offset_dst + y * b.pitch,
                    uv_src + y * src_w, src_w);
      }
    }

    // Compute source crop + on-screen rect based on the configured fit
    // mode. The DRM plane scaler can crop the source (src_x/src_y/src_w/src_h
    // in 16.16 fixed-point) and project to any rect on the CRTC at zero
    // CPU cost, so all three modes are pure metadata changes.
    int crtc_x = 0, crtc_y = 0;
    int crtc_w = screen_w_, crtc_h = screen_h_;
    int sx = 0, sy = 0, sw = src_w, sh = src_h;
    computeFitGeometry(src_w, src_h, crtc_x, crtc_y, crtc_w, crtc_h,
                       sx, sy, sw, sh);

    if (drmModeSetPlane(fd_, plane_id_, crtc_id_, b.fb_id, 0,
                        crtc_x, crtc_y, crtc_w, crtc_h,
                        sx << 16, sy << 16, sw << 16, sh << 16) < 0) {
      std::cerr << "[drm] SetPlane failed: " << std::strerror(errno) << "\n";
    }
  }

  // Three fit modes (env BOARD_LOOPBACK_VIDEO_FIT, default = fill):
  //
  //   fill    aspect-fill: crop a screen-aspect strip from the source's
  //           centre, project to fullscreen. Source covers the whole
  //           screen with no black bars; the cropped edges are lost.
  //           Best for video calls — face usually in the centre.
  //
  //   fit     aspect-fit (letterbox): scale the full source to the
  //           largest rect that fits the screen with the source's
  //           aspect, leaving black bars on the unused edges. Whatever
  //           was on the primary plane shows through the bars — without
  //           a compositor the bars may not be uniformly black, but the
  //           video itself isn't stretched.
  //
  //   stretch source goes fullscreen ignoring aspect (the historic
  //           behaviour). Easy A/B comparison or for any caller who
  //           knows the source is already the right aspect.
  enum class FitMode { Fill, Fit, Stretch };
  FitMode fitMode() const {
    static const FitMode m = []() {
      const char *e = std::getenv("BOARD_LOOPBACK_VIDEO_FIT");
      if (!e) return FitMode::Fill;
      const std::string s(e);
      if (s == "fit") return FitMode::Fit;
      if (s == "stretch") return FitMode::Stretch;
      return FitMode::Fill;
    }();
    return m;
  }

  void computeFitGeometry(int src_w, int src_h,
                          int &crtc_x, int &crtc_y,
                          int &crtc_w, int &crtc_h,
                          int &sx, int &sy, int &sw, int &sh) const {
    crtc_x = 0;
    crtc_y = 0;
    crtc_w = screen_w_;
    crtc_h = screen_h_;
    sx = 0;
    sy = 0;
    sw = src_w;
    sh = src_h;

    if (fitMode() == FitMode::Stretch) {
      return; // historic full-stretch
    }

    // Use 64-bit math to avoid overflow on the cross-multiply comparison
    // (1920×1080×N can blow int32 in the next product).
    const std::int64_t src_a = static_cast<std::int64_t>(src_w) * screen_h_;
    const std::int64_t dst_a = static_cast<std::int64_t>(screen_w_) * src_h;

    if (fitMode() == FitMode::Fill) {
      // Crop a screen-aspect rect from the source's centre.
      if (src_a > dst_a) {
        // Source is wider than screen aspect — shrink crop width.
        sw = static_cast<int>(
            (static_cast<std::int64_t>(src_h) * screen_w_) / screen_h_);
        sx = (src_w - sw) / 2;
      } else if (src_a < dst_a) {
        // Source is taller than screen aspect — shrink crop height.
        sh = static_cast<int>(
            (static_cast<std::int64_t>(src_w) * screen_h_) / screen_w_);
        sy = (src_h - sh) / 2;
      }
      // Equal aspect → no crop, fullscreen.
    } else {
      // Fit (letterbox): keep full source, shrink dest rect.
      if (src_a > dst_a) {
        // Source wider — fit width, letterbox top/bottom.
        crtc_h = static_cast<int>(
            (static_cast<std::int64_t>(src_h) * screen_w_) / src_w);
        crtc_y = (screen_h_ - crtc_h) / 2;
      } else if (src_a < dst_a) {
        // Source taller — fit height, pillarbox left/right.
        crtc_w = static_cast<int>(
            (static_cast<std::int64_t>(src_w) * screen_h_) / src_h);
        crtc_x = (screen_w_ - crtc_w) / 2;
      }
    }

    // NV12 (YUV 4:2:0) chroma sub-sampling requires src crop x/y/w/h
    // *and* the on-screen crtc rect to be even — otherwise the U/V
    // planes can't be addressed at the implied sub-pixel and the
    // driver bails with EINVAL on SetPlane. Round x/y down (so we
    // don't widen past the picture) and w/h down to even.
    auto snap_down_even = [](int v) { return v & ~1; };
    sx = snap_down_even(sx);
    sy = snap_down_even(sy);
    sw = snap_down_even(sw);
    sh = snap_down_even(sh);
    crtc_x = snap_down_even(crtc_x);
    crtc_y = snap_down_even(crtc_y);
    crtc_w = snap_down_even(crtc_w);
    crtc_h = snap_down_even(crtc_h);
  }

  // ---- Phase 8.4.5: split-screen display layout ----

  enum class DisplayLayout { Portrait, Landscape };
  DisplayLayout displayLayout() const {
    static const DisplayLayout m = []() {
      const char *e = std::getenv("BOARD_DISPLAY_LAYOUT");
      if (e && std::string(e) == "landscape") return DisplayLayout::Landscape;
      return DisplayLayout::Portrait;
    }();
    return m;
  }

  // Nearest-neighbor NV12 fill-mode blit.
  // Scales src (src_w × src_h) into the (dst_x, dst_y, dst_w, dst_h) rect of
  // a canvas described by (cy, cuv, canvas_pitch, canvas_h).
  // Fill mode: center-crops src to dst aspect ratio before scaling.
  // rotate_cw90=true: treat src as 90° CW rotated (logical dims swap to
  // src_h×src_w); maps dest pixel (lsx,lsy) → actual src (lsy, src_h-1-lsx).
  // Used to correct camera frames that are physically portrait-mounted but
  // output landscape data from the sensor.
  static void scaleNV12Blit(std::uint8_t *cy, std::uint8_t *cuv,
                             int canvas_pitch, int canvas_h,
                             int dst_x, int dst_y, int dst_w, int dst_h,
                             const std::uint8_t *src, int src_w, int src_h,
                             bool rotate_cw90 = false) {
    (void)canvas_h;
    // Logical source dims after optional 90° CW rotation
    const int lsw = rotate_cw90 ? src_h : src_w;
    const int lsh = rotate_cw90 ? src_w : src_h;

    // Fill-mode center-crop in logical source space
    int cx = 0, cy_off = 0, cw = lsw, ch = lsh;
    if (static_cast<std::int64_t>(cw) * dst_h >
        static_cast<std::int64_t>(dst_w) * ch) {
      cw = static_cast<int>(static_cast<std::int64_t>(ch) * dst_w / dst_h);
      cx = (lsw - cw) / 2;
    } else if (static_cast<std::int64_t>(cw) * dst_h <
               static_cast<std::int64_t>(dst_w) * ch) {
      ch = static_cast<int>(static_cast<std::int64_t>(cw) * dst_h / dst_w);
      cy_off = (lsh - ch) / 2;
    }
    cx    &= ~1; cy_off &= ~1; cw &= ~1; ch &= ~1;
    dst_x &= ~1; dst_y  &= ~1; dst_w &= ~1; dst_h &= ~1;

    const std::int64_t y_scale =
        dst_h > 0 ? (static_cast<std::int64_t>(ch) << 16) / dst_h : 0;
    const std::int64_t x_scale =
        dst_w > 0 ? (static_cast<std::int64_t>(cw) << 16) / dst_w : 0;

    if (rotate_cw90) {
      // Y plane — logical (lsx,lsy) → actual (asx=lsy, asy=src_h-1-lsx)
      for (int y = 0; y < dst_h; ++y) {
        const int lsy =
            cy_off + static_cast<int>((static_cast<std::int64_t>(y) * y_scale) >> 16);
        std::uint8_t *dst_row = cy + (dst_y + y) * canvas_pitch + dst_x;
        for (int x = 0; x < dst_w; ++x) {
          const int lsx =
              cx + static_cast<int>((static_cast<std::int64_t>(x) * x_scale) >> 16);
          dst_row[x] = src[(src_h - 1 - lsx) * src_w + lsy];
        }
      }
      // UV plane — logical (lux,luy) → actual (aux=luy, auy=src_h/2-1-lux)
      const std::int64_t uvy_scale =
          (dst_h / 2) > 0
              ? (static_cast<std::int64_t>(ch / 2) << 16) / (dst_h / 2)
              : 0;
      const std::int64_t uvx_scale =
          (dst_w / 2) > 0
              ? (static_cast<std::int64_t>(cw / 2) << 16) / (dst_w / 2)
              : 0;
      const std::uint8_t *src_uv = src + src_w * src_h;
      for (int y = 0; y < dst_h / 2; ++y) {
        const int luy =
            cy_off / 2 +
            static_cast<int>((static_cast<std::int64_t>(y) * uvy_scale) >> 16);
        std::uint8_t *dst_row = cuv + (dst_y / 2 + y) * canvas_pitch + dst_x;
        for (int x = 0; x < dst_w / 2; ++x) {
          const int lux =
              cx / 2 +
              static_cast<int>((static_cast<std::int64_t>(x) * uvx_scale) >> 16);
          const std::uint8_t *sp =
              src_uv + (src_h / 2 - 1 - lux) * src_w + luy * 2;
          dst_row[x * 2]     = sp[0];
          dst_row[x * 2 + 1] = sp[1];
        }
      }
    } else {
      // Y plane (no rotation)
      for (int y = 0; y < dst_h; ++y) {
        const int sy =
            cy_off + static_cast<int>((static_cast<std::int64_t>(y) * y_scale) >> 16);
        const std::uint8_t *src_row = src + sy * src_w;
        std::uint8_t       *dst_row = cy + (dst_y + y) * canvas_pitch + dst_x;
        for (int x = 0; x < dst_w; ++x) {
          const int sx =
              cx + static_cast<int>((static_cast<std::int64_t>(x) * x_scale) >> 16);
          dst_row[x] = src_row[sx];
        }
      }
      // UV plane (no rotation)
      const std::int64_t uvy_scale =
          (dst_h / 2) > 0
              ? (static_cast<std::int64_t>(ch / 2) << 16) / (dst_h / 2)
              : 0;
      const std::int64_t uvx_scale =
          (dst_w / 2) > 0
              ? (static_cast<std::int64_t>(cw / 2) << 16) / (dst_w / 2)
              : 0;
      const std::uint8_t *src_uv = src + src_w * src_h;
      for (int y = 0; y < dst_h / 2; ++y) {
        const int sy =
            cy_off / 2 +
            static_cast<int>((static_cast<std::int64_t>(y) * uvy_scale) >> 16);
        const std::uint8_t *src_row = src_uv + sy * src_w;
        std::uint8_t       *dst_row =
            cuv + (dst_y / 2 + y) * canvas_pitch + dst_x;
        for (int x = 0; x < dst_w / 2; ++x) {
          const int sx =
              cx / 2 +
              static_cast<int>((static_cast<std::int64_t>(x) * uvx_scale) >> 16);
          dst_row[x * 2]     = src_row[sx * 2];
          dst_row[x * 2 + 1] = src_row[sx * 2 + 1];
        }
      }
    }
  }

  // Full-screen CPU-rotated preview for local camera frames (portrait-mounted
  // sensor outputting landscape data). Uses composite buffers so DRM plane
  // receives a pre-rotated NV12 frame at screen resolution.
  void renderNV12Rotated(const std::uint8_t *src, int src_w, int src_h) {
    if (src_w < 8 || src_h < 8) return;
    std::lock_guard<std::mutex> render_lock(render_mutex_);
    if (!ensureCompositeBufs()) return;
    DumbBuf &b = cbufs_[comp_next_];
    comp_next_ ^= 1;

    const std::size_t y_size = static_cast<std::size_t>(b.pitch) * screen_h_;
    std::uint8_t *const cy_plane  = b.mapped;
    std::uint8_t *const cuv_plane = b.mapped + y_size;
    std::memset(cy_plane,  0x10, y_size);
    std::memset(cuv_plane, 0x80, y_size / 2);

    scaleNV12Blit(cy_plane, cuv_plane, b.pitch, screen_h_,
                  0, 0, screen_w_, screen_h_,
                  src, src_w, src_h, /*rotate_cw90=*/true);

    if (drmModeSetPlane(fd_, plane_id_, crtc_id_, b.fb_id, 0,
                        0, 0, screen_w_, screen_h_,
                        0, 0, screen_w_ << 16, screen_h_ << 16) < 0) {
      std::cerr << "[drm] SetPlane (rotated preview) failed: "
                << std::strerror(errno) << "\n";
    }
  }

  // Allocate a pair of screen-sized dumb buffers for compositing.
  bool ensureCompositeBufs() {
    if (comp_w_ == screen_w_ && comp_h_ == screen_h_) return true;
    freeCompositeBufs();
    comp_w_ = screen_w_;
    comp_h_ = screen_h_;
    for (int i = 0; i < 2; ++i) {
      DumbBuf &b = cbufs_[i];
      drm_mode_create_dumb cd{};
      cd.width  = static_cast<std::uint32_t>(screen_w_);
      cd.height = static_cast<std::uint32_t>(screen_h_ * 3 / 2);
      cd.bpp    = 8;
      if (drmIoctl(fd_, DRM_IOCTL_MODE_CREATE_DUMB, &cd) < 0) {
        std::cerr << "[drm] composite CREATE_DUMB failed: "
                  << std::strerror(errno) << "\n";
        return false;
      }
      b.handle = cd.handle;
      b.size   = cd.size;
      b.pitch  = cd.pitch;
      drm_mode_map_dumb md{};
      md.handle = b.handle;
      if (drmIoctl(fd_, DRM_IOCTL_MODE_MAP_DUMB, &md) < 0) {
        std::cerr << "[drm] composite MAP_DUMB failed: "
                  << std::strerror(errno) << "\n";
        return false;
      }
      b.mapped = static_cast<std::uint8_t *>(
          mmap(nullptr, b.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_,
               md.offset));
      if (b.mapped == MAP_FAILED) {
        std::cerr << "[drm] composite mmap failed: "
                  << std::strerror(errno) << "\n";
        return false;
      }
      std::uint32_t handles[4] = {b.handle, b.handle, 0, 0};
      std::uint32_t pitches[4] = {b.pitch, b.pitch, 0, 0};
      std::uint32_t offsets[4] = {
          0,
          b.pitch * static_cast<std::uint32_t>(screen_h_),
          0, 0};
      if (drmModeAddFB2(fd_, screen_w_, screen_h_, DRM_FORMAT_NV12, handles,
                        pitches, offsets, &b.fb_id, 0) < 0) {
        std::cerr << "[drm] composite AddFB2 failed: "
                  << std::strerror(errno) << "\n";
        return false;
      }
    }
    std::cout << "[drm] composite buffers ready: 2x " << screen_w_ << "x"
              << screen_h_ << " NV12\n";
    return true;
  }

  // Composite remote + local NV12 frames into one screen-sized buffer and
  // display it via DRM plane.
  // Portrait layout (default): top half = remote, bottom half = local.
  // Landscape layout: left half = remote, right half = local.
  void renderSplitNV12(const std::uint8_t *remote, int rw, int rh,
                       const std::uint8_t *local,  int lw, int lh) {
    if (rw < 8 || rh < 8 || lw < 8 || lh < 8) return;
    std::lock_guard<std::mutex> render_lock(render_mutex_);
    if (!ensureCompositeBufs()) return;
    DumbBuf &b = cbufs_[comp_next_];
    comp_next_ ^= 1;

    const std::size_t y_size = static_cast<std::size_t>(b.pitch) * screen_h_;
    std::uint8_t *const cy  = b.mapped;
    std::uint8_t *const cuv = b.mapped + y_size;
    // BT.601 limited-range black: Y=16, UV=128
    std::memset(cy,  0x10, y_size);
    std::memset(cuv, 0x80, y_size / 2);

    if (displayLayout() == DisplayLayout::Portrait) {
      const int half_h = screen_h_ / 2;
      scaleNV12Blit(cy, cuv, b.pitch, screen_h_,
                    0, 0,      screen_w_, half_h, remote, rw, rh);
      scaleNV12Blit(cy, cuv, b.pitch, screen_h_,
                    0, half_h, screen_w_, half_h, local,  lw, lh,
                    /*rotate_cw90=*/true);
    } else {
      const int half_w = screen_w_ / 2;
      scaleNV12Blit(cy, cuv, b.pitch, screen_h_,
                    0,      0, half_w, screen_h_, remote, rw, rh);
      scaleNV12Blit(cy, cuv, b.pitch, screen_h_,
                    half_w, 0, half_w, screen_h_, local,  lw, lh,
                    /*rotate_cw90=*/true);
    }

    if (drmModeSetPlane(fd_, plane_id_, crtc_id_, b.fb_id, 0,
                        0, 0, screen_w_, screen_h_,
                        0, 0,
                        screen_w_ << 16, screen_h_ << 16) < 0) {
      std::cerr << "[drm] SetPlane (split) failed: "
                << std::strerror(errno) << "\n";
    }
  }

  ~DrmDisplay() { close(); }

  void close() {
    freeBuffers();
    freeCompositeBufs();
    if (fd_ >= 0) {
      drmDropMaster(fd_);
      ::close(fd_);
      fd_ = -1;
    }
  }

private:
  int crtcIndex() {
    drmModeRes *res = drmModeGetResources(fd_);
    int idx = 0;
    for (int i = 0; i < res->count_crtcs; ++i) {
      if (res->crtcs[i] == crtc_id_) {
        idx = i;
        break;
      }
    }
    drmModeFreeResources(res);
    return idx;
  }

  void freeBuffers() {
    for (auto &b : bufs_) {
      if (b.fb_id) {
        drmModeRmFB(fd_, b.fb_id);
        b.fb_id = 0;
      }
      if (b.mapped && b.mapped != MAP_FAILED) {
        munmap(b.mapped, b.size);
        b.mapped = nullptr;
      }
      if (b.handle) {
        drm_mode_destroy_dumb dd{};
        dd.handle = b.handle;
        drmIoctl(fd_, DRM_IOCTL_MODE_DESTROY_DUMB, &dd);
        b.handle = 0;
      }
    }
    buf_w_ = buf_h_ = 0;
  }

  void freeCompositeBufs() {
    for (auto &b : cbufs_) {
      if (b.fb_id) {
        drmModeRmFB(fd_, b.fb_id);
        b.fb_id = 0;
      }
      if (b.mapped && b.mapped != MAP_FAILED) {
        munmap(b.mapped, b.size);
        b.mapped = nullptr;
      }
      if (b.handle) {
        drm_mode_destroy_dumb dd{};
        dd.handle = b.handle;
        drmIoctl(fd_, DRM_IOCTL_MODE_DESTROY_DUMB, &dd);
        b.handle = 0;
      }
    }
    comp_w_ = comp_h_ = 0;
  }

  struct DumbBuf {
    std::uint32_t handle = 0;
    std::uint32_t fb_id = 0;
    std::uint32_t pitch = 0;
    std::uint64_t size = 0;
    std::uint8_t *mapped = nullptr;
  };

  int fd_ = -1;
  std::uint32_t conn_id_ = 0;
  std::uint32_t crtc_id_ = 0;
  std::uint32_t plane_id_ = 0;
  drmModeModeInfo mode_{};
  int screen_w_ = 0;
  int screen_h_ = 0;
  DumbBuf bufs_[2];
  int next_ = 0;
  int buf_w_ = 0;
  int buf_h_ = 0;
  // Phase 8.4.5: screen-sized composite buffers for split-screen rendering
  DumbBuf cbufs_[2];
  int comp_next_ = 0;
  int comp_w_    = 0;
  int comp_h_    = 0;
  // Phase 8.4.1：保护 ensureBuffers 的 free/realloc 临界区，防多 peer
  // 并发 callback use-after-free。
  std::mutex render_mutex_;
};

static DrmDisplay g_drm;
static std::atomic<bool> g_drm_ok{false};

// Phase 8.4.5: local camera frame stash (updated every V4L2 frame, read by
// VideoRouter to composite the split-screen local half).
static std::atomic<bool>    g_in_meeting{false};
static std::mutex            g_local_frame_mu;
static std::vector<std::uint8_t> g_local_frame_data;
static int g_local_frame_w = 0;
static int g_local_frame_h = 0;

// VideoRouter — Phase 8.4.1
// ---------------------------------------------------------------
// 多 peer 时 BoardLoopback 是单显示器 = 同一时间只能放一路远端视频。
// 每个 peer 的 video track callback 进来都问 router："我是 active 吗？"
// 不是 active 直接 return，是 active 才进 DrmDisplay 临界区渲染。
//
// 8.4.1 MVP 行为：第一个 peer 自动成为 active；setActive 可手动切换
// （8.4.2 会接到 onActiveSpeakersChanged 自动切换）。peer 离开时
// resetIfActive 清空 active_id，下一个 peer frame 接管。
class VideoRouter {
public:
  void renderIfActive(const std::string &id, const std::uint8_t *data,
                      int w, int h) {
    bool match;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (active_id_.empty()) {
        active_id_ = id;
        std::cout << "[router] first peer wins active: " << id << "\n";
      }
      match = (id == active_id_);
    }
    if (!match)
      return;
    // Phase 8.4.5: split-screen — composite remote + local frames
    if (g_in_meeting.load(std::memory_order_relaxed)) {
      std::vector<std::uint8_t> local_copy;
      int lw = 0, lh = 0;
      {
        std::lock_guard<std::mutex> lk(g_local_frame_mu);
        if (!g_local_frame_data.empty()) {
          local_copy = g_local_frame_data;
          lw = g_local_frame_w;
          lh = g_local_frame_h;
        }
      }
      if (lw > 0 && lh > 0)
        g_drm.renderSplitNV12(data, w, h, local_copy.data(), lw, lh);
      else
        g_drm.renderNV12(data, w, h); // fallback: no local frame yet
    } else {
      g_drm.renderNV12(data, w, h);
    }
  }

  // 设置 active peer id；空字符串表示无 active（下一个进来的 peer 会
  // 自动接管）。8.4.2 active speaker handler 会从这里切。
  void setActive(const std::string &id) {
    std::lock_guard<std::mutex> lk(mu_);
    if (active_id_ != id) {
      std::cout << "[router] active: '" << active_id_ << "' -> '" << id
                << "'\n";
      active_id_ = id;
    }
  }

  // peer 离开时调用：如果它是当前 active，清空让下一个 frame 接管。
  void resetIfActive(const std::string &id) {
    std::lock_guard<std::mutex> lk(mu_);
    if (active_id_ == id) {
      std::cout << "[router] active peer '" << id << "' left, clearing\n";
      active_id_.clear();
    }
  }

  std::string getActive() const {
    std::lock_guard<std::mutex> lk(mu_);
    return active_id_;
  }

private:
  mutable std::mutex mu_;
  std::string active_id_;
};

static VideoRouter g_video_router;

// MeetingController — Phase 8.4.2 (MVP + 完整 hold-down)
// ---------------------------------------------------------------
// 单 worker thread + 命令队列，串行处理 active speaker 切换 + 视频订阅
// 控制。所有 SDK delegate 事件都投递 Cmd 进队列，worker 串行 reconcile，
// 避免多 delegate 线程并发改 setSubscribed / VideoRouter active 状态。
//
// 行为：
//   - onActiveSpeakersChanged → ActiveSpeakerObserved cmd（仅"观察"，不立即切）
//   - onTrackPublished(VIDEO) → TrackPublishedDefend cmd（非 active 立即 unsub
//     防 10 peer 全订阅解码爆 CPU）
//   - 切换需要新 speaker 连续观察 ≥ kHoldDownMs 才 commit（默认 600ms，可通过
//     BOARD_ACTIVE_SPEAKER_HOLDDOWN_MS 调）。worker 在 cv.wait_until 上挂 deadline，
//     到点检查 pending 是否仍然有效，是的话 commit (sub new → router → unsub old)
//
// 为什么需要 hold-down（2026-04-29 实测）：
// 手机 + Web 同时讲话时，SDK 的 onActiveSpeakersChanged 在两者间 < 500ms 反复
// 抖动；MVP 直接切导致 sub/unsub 风暴 → 在 liblivekit_ffi 内部 malloc 报
// "corrupted double-linked list" SIGABRT（FFI 侧 race，我们改不了）。600ms
// 的 hold-down 足以让"同时讲话"自然收敛到稳定 speaker，避免高频 churn。
class MeetingController {
public:
  enum class CmdType {
    ActiveSpeakerObserved, // identity = 新 active 的 id
    TrackPublishedDefend,  // identity = pub owner，sid = pub.sid
    Shutdown,
  };
  struct Cmd {
    CmdType type;
    std::string identity;
    std::string sid;
  };

  MeetingController(Room *room, std::string local_identity)
      : room_(room), local_identity_(std::move(local_identity)) {
    if (const char *e = std::getenv("BOARD_ACTIVE_SPEAKER_HOLDDOWN_MS")) {
      try {
        int ms = std::stoi(e);
        if (ms >= 0 && ms < 60000) hold_down_ms_ = ms;
      } catch (...) { /* keep default */ }
    }
    worker_ = std::thread([this] { workerLoop(); });
  }

  ~MeetingController() {
    enqueue({CmdType::Shutdown, "", ""});
    if (worker_.joinable())
      worker_.join();
  }

  void enqueue(Cmd cmd) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push_back(std::move(cmd));
    }
    cv_.notify_one();
  }

  // 直接给 LoopbackDelegate / 其它地方读：当前 active 是谁（仅供日志/UI 用）。
  std::string currentActive() const {
    std::lock_guard<std::mutex> lk(state_mu_);
    return current_active_;
  }

private:
  using Clock = std::chrono::steady_clock;

  void workerLoop() {
    while (true) {
      Cmd cmd;
      bool got_cmd = false;
      {
        std::unique_lock<std::mutex> lk(mu_);
        // 计算下一个 hold-down deadline；没 pending 就长睡到下次 enqueue
        // notify。命令队列非空也立刻醒，正常 reconcile。
        auto deadline = Clock::now() + std::chrono::hours(1);
        {
          std::lock_guard<std::mutex> sl(state_mu_);
          if (!pending_active_.empty()) {
            deadline = pending_first_seen_ +
                       std::chrono::milliseconds(hold_down_ms_);
          }
        }
        cv_.wait_until(lk, deadline, [this] { return !queue_.empty(); });
        if (!queue_.empty()) {
          cmd = std::move(queue_.front());
          queue_.pop_front();
          got_cmd = true;
        }
      }
      if (got_cmd) {
        switch (cmd.type) {
        case CmdType::Shutdown:
          return;
        case CmdType::ActiveSpeakerObserved:
          observeActiveSpeaker(cmd.identity);
          break;
        case CmdType::TrackPublishedDefend:
          handleTrackPublishedDefend(cmd.identity, cmd.sid);
          break;
        }
      }
      // 不论是 cmd 处理完还是 deadline 到，都检查一次 pending 是否该 commit
      tryCommitPendingActive();
    }
  }

  // observeActiveSpeaker：仅记录"观察到 X"，等 hold-down 到点再 commit。
  // 同 X 重复观察刷新时间戳没意义（不重置 first_seen），不同 X 就把 pending
  // 换成新 X，重置 first_seen 倒数。
  void observeActiveSpeaker(const std::string &observed) {
    if (observed.empty())
      return;
    std::lock_guard<std::mutex> lk(state_mu_);
    if (observed == current_active_) {
      // 已经是 active，把 pending 清掉（防之前另一个 candidate 卡在 pending）
      pending_active_.clear();
      return;
    }
    if (observed == pending_active_) {
      // 同一个候选仍在被观察 — 不重置 first_seen，让 hold-down 自然到期
      return;
    }
    // 新候选 / candidate 切换 → 重置 hold-down 倒数
    pending_active_ = observed;
    pending_first_seen_ = Clock::now();
  }

  // tryCommitPendingActive：worker 醒来时调（每次 cmd 处理或 timeout 后）。
  // 检查 pending 是否还在、且已经过 hold-down 时间，是的话 commit 切换。
  void tryCommitPendingActive() {
    std::string new_active, old_active;
    {
      std::lock_guard<std::mutex> lk(state_mu_);
      if (pending_active_.empty())
        return;
      auto elapsed = Clock::now() - pending_first_seen_;
      if (elapsed < std::chrono::milliseconds(hold_down_ms_))
        return; // 还没过 hold-down
      if (pending_active_ == current_active_) {
        pending_active_.clear();
        return; // race：变成 current 了
      }
      new_active = pending_active_;
      old_active = current_active_;
      current_active_ = new_active;
      pending_active_.clear();
    }
    commitActiveSpeakerSwitch(old_active, new_active);
  }

  // Phase 8.4.2 (post-incident 2026-04-29): 切换只动 VideoRouter，**不再
  // 调 setSubscribed(false) 退订旧 active 视频**。所有远端视频都保持
  // subscribed，靠 router 切渲染。
  //
  // 为什么：实测 hold-down 600ms 仍然崩 — `corrupted double-linked list`
  // SIGABRT 在 liblivekit_ffi 内 realloc 路径。每次 unsub 都会让 FFI 累积
  // heap corruption（不是单纯的 churn 频率问题）。SDK 这个 race 我们改不
  // 了，只能不触发。代价：N peer 全 stay-subscribed → N 路 HW 解码（MPP
  // 实测 2-3 路无压力；10 路需先修 SDK bug 再优化）。
  void commitActiveSpeakerSwitch(const std::string &old_active,
                                 const std::string &new_active) {
    std::cout << "[ctrl] active speaker: '" << old_active << "' -> '"
              << new_active << "' (hold-down " << hold_down_ms_
              << "ms, render-only switch)\n";

    // 确保新 active 是 subscribed（应该一直都是 subscribed，但保险起见）
    if (auto *p = room_->remoteParticipant(new_active)) {
      for (auto &kv : p->trackPublications()) {
        const auto &pub = kv.second;
        if (pub && pub->kind() == TrackKind::KIND_VIDEO) {
          if (!pub->subscribed()) {
            pub->setSubscribed(true);
            std::cout << "[ctrl] sub video: " << new_active << " sid="
                      << kv.first << "\n";
          }
        }
      }
    }

    // 切 router 显示源（atomic，纯 C++ 状态变化，不进 FFI）
    g_video_router.setActive(new_active);
  }

  // Phase 8.4.2 (post-incident 2026-04-29): defend-unsub 也禁用 — 同样会
  // 触发 liblivekit_ffi heap corruption。所有 peer video 留 subscribed。
  // 10 peer 解码 CPU 风险 deferred 到 SDK FFI bug 修后再处理。
  void handleTrackPublishedDefend(const std::string & /*identity*/,
                                  const std::string & /*sid*/) {
    // no-op: 见 commitActiveSpeakerSwitch 的注释
  }

  Room *room_;
  std::string local_identity_;
  std::thread worker_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Cmd> queue_;

  mutable std::mutex state_mu_;
  std::string current_active_;            // protected by state_mu_
  std::string pending_active_;            // 候选切换目标，hold-down 期间累积
  Clock::time_point pending_first_seen_;  // 候选首次被观察的时间
  int hold_down_ms_ = 600;                // 默认 600ms，可 env 调
};

static std::unique_ptr<MeetingController> g_meeting_controller;

// MeetingState — Phase 8.4.4
// ---------------------------------------------------------------
// 进程级会议生命周期状态机 + HTTP join 命令 inbox。HTTP /v1/meeting/join
// 把 (url, token) 投到这里 → main thread 在 waitForJoin 上 cv 等到后
// 取出参数进入 runMeeting()。HTTP /v1/meeting/leave 触发 leaveRequested
// flag → 主循环检查到后退出 → cleanup → 进程退出（MVP 一会议一进程）。
class MeetingState {
public:
  // Phase 8.4.4-完整：daemon 模式新增 Shutdown 状态。signal handler 投
  // requestShutdown() 让 waitForJoinOrShutdown 返回 nullopt，外循环 break
  // 退 daemon。/leave 只把 InMeeting → Leaving，回 Idle 等下一会。
  enum class State { Idle, Joining, InMeeting, Leaving, Shutdown };

  // HTTP /join：成功返回 {true, ""}，失败返回 {false, "<err>"}
  std::pair<bool, std::string> requestJoin(std::string url,
                                            std::string token) {
    std::lock_guard<std::mutex> lk(mu_);
    if (state_ == State::Shutdown) {
      return {false, "daemon shutting down"};
    }
    if (state_ != State::Idle) {
      return {false, "already in/past meeting; wait for /leave then retry"};
    }
    if (url.empty() || token.empty()) {
      return {false, "url/token must be non-empty"};
    }
    pending_url_ = std::move(url);
    pending_token_ = std::move(token);
    state_ = State::Joining;
    cv_.notify_all();
    return {true, ""};
  }

  // HTTP /leave：成功返回 true，状态不对返回 false
  bool requestLeave() {
    std::lock_guard<std::mutex> lk(mu_);
    if (state_ != State::InMeeting)
      return false;
    state_ = State::Leaving;
    return true;
  }

  // signal handler 调：daemon 收到 SIGINT/SIGTERM 进 Shutdown，
  // 唤醒等在 waitForJoinOrShutdown 的 main thread 让它 break 外循环。
  void requestShutdown() {
    std::lock_guard<std::mutex> lk(mu_);
    state_ = State::Shutdown;
    cv_.notify_all();
  }

  // main thread：阻塞直到 join 触发或 shutdown。返回 nullopt 表示
  // 收到 shutdown 应退出 daemon 外循环。
  std::optional<std::pair<std::string, std::string>>
  waitForJoinOrShutdown() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] {
      return state_ == State::Joining || state_ == State::Shutdown;
    });
    if (state_ == State::Shutdown)
      return std::nullopt;
    return std::make_pair(pending_url_, pending_token_);
  }

  // runMeeting() 接通 Room 后进 InMeeting 状态，记下 room_name 给 status
  void setInMeeting(const std::string &room_name, Room *room) {
    std::lock_guard<std::mutex> lk(mu_);
    if (state_ == State::Shutdown)
      return; // 在 shutdown 中就别再进 InMeeting 了
    state_ = State::InMeeting;
    room_name_ = room_name;
    room_for_status_.store(room, std::memory_order_release);
  }

  // 会议结束清状态。守护 Shutdown 态（防止外循环误以为还要继续）。
  void clearMeeting() {
    std::lock_guard<std::mutex> lk(mu_);
    if (state_ != State::Shutdown) {
      state_ = State::Idle;
    }
    room_name_.clear();
    pending_url_.clear();
    pending_token_.clear();
    room_for_status_.store(nullptr, std::memory_order_release);
  }

  bool isShutdown() const {
    std::lock_guard<std::mutex> lk(mu_);
    return state_ == State::Shutdown;
  }

  struct Snapshot {
    std::string state;
    std::string room_name;
    std::string active_speaker;
    int participant_count = 0;
  };

  Snapshot snapshot() const {
    Snapshot s;
    {
      std::lock_guard<std::mutex> lk(mu_);
      switch (state_) {
      case State::Idle:
        s.state = "idle";
        break;
      case State::Joining:
        s.state = "joining";
        break;
      case State::InMeeting:
        s.state = "in_meeting";
        break;
      case State::Leaving:
        s.state = "leaving";
        break;
      case State::Shutdown:
        s.state = "shutdown";
        break;
      }
      s.room_name = room_name_;
    }
    if (g_meeting_controller)
      s.active_speaker = g_meeting_controller->currentActive();
    if (auto *room = room_for_status_.load(std::memory_order_acquire))
      s.participant_count =
          static_cast<int>(room->remoteParticipants().size());
    return s;
  }

private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  State state_ = State::Idle;
  std::string pending_url_;
  std::string pending_token_;
  std::string room_name_;
  std::atomic<Room *> room_for_status_{nullptr};
};

static MeetingState g_meeting_state;

// Phase 8.4.4-完整: 现在 g_meeting_state 完全声明好，可以给 handleSignal
// 提供真正的实现。daemon 模式下 SIGINT/SIGTERM 必须唤醒等在
// waitForJoinOrShutdown 上的 main thread 才能让外循环 break 退进程。
//
// 注意：std::lock_guard + cv.notify 在 signal handler 严格说不 async-signal-
// safe（pthread_mutex_lock 不在 POSIX async-safe 列表上）。实践中 mutex 持
// 锁极短不死锁，但严格场景应换 self-pipe trick；留给后续改进。
static void handleSignal(int) {
  g_running.store(false);
  g_meeting_state.requestShutdown();
}

// Minimal JSON helpers — Phase 8.4.4
// ---------------------------------------------------------------
// 我们对 JSON 的需求极小：解析 `{"url":"...","token":"..."}` 取两个字符串
// 字段，生成 `{"key":"value", ...}` 几个字段的响应。引一个 nlohmann_json
// 仅为这点用例不值得（Buildroot 不带，FetchContent 网络又卡），手写。
namespace mini_json {

// 字符串 escape：双引号、反斜杠、控制字符。够生产 token / url / identity
// 这些 ASCII 友好场景；不处理 \uXXXX。
inline std::string escape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 4);
  for (char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
        out += buf;
      } else {
        out += c;
      }
    }
  }
  return out;
}

// 解析 JSON 字符串：从 `s` 第 `pos` 位（指向开 `"`）开始，返回 (value, end_pos)。
// 失败抛 runtime_error。
inline std::pair<std::string, std::size_t> parseString(const std::string &s,
                                                       std::size_t pos) {
  if (pos >= s.size() || s[pos] != '"')
    throw std::runtime_error("expected '\"' at offset " + std::to_string(pos));
  std::string out;
  ++pos;
  while (pos < s.size()) {
    char c = s[pos];
    if (c == '"')
      return {out, pos + 1};
    if (c == '\\' && pos + 1 < s.size()) {
      char e = s[pos + 1];
      switch (e) {
      case '"':
        out += '"';
        break;
      case '\\':
        out += '\\';
        break;
      case '/':
        out += '/';
        break;
      case 'b':
        out += '\b';
        break;
      case 'f':
        out += '\f';
        break;
      case 'n':
        out += '\n';
        break;
      case 'r':
        out += '\r';
        break;
      case 't':
        out += '\t';
        break;
      default:
        // 不支持 \uXXXX；token / url 不会用
        throw std::runtime_error("unsupported escape \\" + std::string(1, e));
      }
      pos += 2;
    } else {
      out += c;
      ++pos;
    }
  }
  throw std::runtime_error("unterminated string");
}

// 在简单 flat object `{"k1":"v1","k2":"v2"}` 里取 key 的字符串值。
// 不支持嵌套 / 数组 / 数字 —— 我们的 schema 是 (url, token) 两字符串字段，够用。
// 找不到 key 返回空 optional。
inline std::optional<std::string> getString(const std::string &json,
                                             const std::string &key) {
  // 找 `"key"` 子串
  const std::string needle = "\"" + key + "\"";
  std::size_t pos = json.find(needle);
  if (pos == std::string::npos)
    return std::nullopt;
  pos += needle.size();
  // skip whitespace + ':' + whitespace
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos])))
    ++pos;
  if (pos >= json.size() || json[pos] != ':')
    return std::nullopt;
  ++pos;
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos])))
    ++pos;
  try {
    auto [val, _] = parseString(json, pos);
    return val;
  } catch (...) {
    return std::nullopt;
  }
}

} // namespace mini_json

// BoardHttpServer — Phase 8.4.4
// ---------------------------------------------------------------
// cpp-httplib 包装。listen_thread_ 跑 server.listen 阻塞；server.stop()
// 让它返回，然后 join。所有 POST 必须带 Authorization: Bearer <BOARD_API_TOKEN>，
// GET /status 不要 auth（运维诊断用）。端点：
//   POST /v1/meeting/join  body {"url":"wss://...","token":"<jwt>"} → 202 / 401 / 409 / 400
//   POST /v1/meeting/leave                                          → 200 / 401 / 409
//   GET  /v1/meeting/status                                         → 200 (always)
class BoardHttpServer {
public:
  BoardHttpServer(std::string bearer, int port)
      : bearer_(std::move(bearer)), port_(port) {
    setupRoutes();
  }

  void start() {
    listen_thread_ = std::thread([this] {
      std::cout << "[http] listening on 0.0.0.0:" << port_ << "\n";
      // listen 阻塞直到 server.stop()
      if (!server_.listen("0.0.0.0", port_)) {
        std::cerr << "[http] listen on 0.0.0.0:" << port_ << " failed\n";
      }
    });
    // 等 server 进入 listen 循环再返回（avoid race in tests）
    int waited_ms = 0;
    while (!server_.is_running() && waited_ms < 2000) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      waited_ms += 20;
    }
  }

  void stop() {
    server_.stop();
    if (listen_thread_.joinable())
      listen_thread_.join();
  }

private:
  bool checkBearer(const httplib::Request &req) const {
    auto it = req.headers.find("Authorization");
    if (it == req.headers.end())
      return false;
    const std::string expected = "Bearer " + bearer_;
    return it->second == expected;
  }

  static void reject401(httplib::Response &res) {
    res.status = 401;
    res.set_content(R"({"error":"unauthorized: missing or invalid bearer"})",
                    "application/json");
  }

  void setupRoutes() {
    server_.Post("/v1/meeting/join", [this](const httplib::Request &req,
                                            httplib::Response &res) {
      if (!checkBearer(req)) {
        reject401(res);
        return;
      }
      auto url_opt = mini_json::getString(req.body, "url");
      auto token_opt = mini_json::getString(req.body, "token");
      if (!url_opt || !token_opt) {
        res.status = 400;
        res.set_content(
            R"({"error":"bad json: expected {\"url\":\"...\",\"token\":\"...\"}"})",
            "application/json");
        return;
      }
      auto [ok, err] = g_meeting_state.requestJoin(std::move(*url_opt),
                                                    std::move(*token_opt));
      if (ok) {
        res.status = 202;
        res.set_content(R"({"status":"joining"})", "application/json");
      } else {
        res.status = 409;
        std::string body = R"({"error":")" + mini_json::escape(err) + "\"}";
        res.set_content(body, "application/json");
      }
    });

    server_.Post("/v1/meeting/leave", [this](const httplib::Request &req,
                                             httplib::Response &res) {
      if (!checkBearer(req)) {
        reject401(res);
        return;
      }
      if (g_meeting_state.requestLeave()) {
        // 让 main loop 退出（runMeeting 检查 g_running）
        g_running.store(false);
        res.status = 200;
        res.set_content(R"({"status":"leaving"})", "application/json");
      } else {
        res.status = 409;
        res.set_content(R"({"error":"not in meeting"})", "application/json");
      }
    });

    server_.Get("/v1/meeting/status",
                [](const httplib::Request &, httplib::Response &res) {
                  auto snap = g_meeting_state.snapshot();
                  std::string body = "{";
                  body += "\"state\":\"" + mini_json::escape(snap.state) + "\",";
                  body += "\"room_name\":\"" +
                          mini_json::escape(snap.room_name) + "\",";
                  body += "\"active_speaker\":\"" +
                          mini_json::escape(snap.active_speaker) + "\",";
                  body += "\"participant_count\":" +
                          std::to_string(snap.participant_count);
                  body += "}";
                  res.status = 200;
                  res.set_content(body, "application/json");
                });
  }

  httplib::Server server_;
  std::string bearer_;
  int port_;
  std::thread listen_thread_;
};

class LoopbackDelegate : public RoomDelegate {
public:
  void onParticipantConnected(Room &,
                              const ParticipantConnectedEvent &ev) override {
    if (ev.participant) {
      std::cout << "[loopback] participant connected: "
                << ev.participant->identity() << "\n";
    }
  }

  void onParticipantDisconnected(
      Room &, const ParticipantDisconnectedEvent &ev) override {
    if (ev.participant) {
      const std::string id = ev.participant->identity();
      std::cout << "[loopback] participant disconnected: " << id << "\n";
      // Phase 8.4.1: 如果离开的是当前 active 渲染的 peer，清空 router；
      // 下一个还在房间里的 peer 的下一帧到来时自动接管。
      g_video_router.resetIfActive(id);
      // Phase 8.4.3: 清掉这个 peer 的 audio ring buffer，省内存（10 peer
      // × 100ms ring = ~9.6KB，无所谓但语义干净）。
      g_mixer.removePeer(id);
    }
  }

  // Phase 8.4.2: 远端发布 track 通知。video track 进来如果不是当前 active，
  // 立刻让 controller 来 unsub，省 CPU；audio track 不动（任何时候都要订阅，
  // 8.4.3 软混音才有数据可混）。
  void onTrackPublished(Room &, const TrackPublishedEvent &ev) override {
    if (!ev.participant || !ev.publication)
      return;
    const std::string identity = ev.participant->identity();
    const std::string sid = ev.publication->sid();
    const TrackKind kind = ev.publication->kind();
    std::cout << "[loopback] track published: " << identity << " kind="
              << (kind == TrackKind::KIND_VIDEO ? "video"
                  : kind == TrackKind::KIND_AUDIO ? "audio"
                                                  : "unknown")
              << " sid=" << sid << "\n";
    if (kind == TrackKind::KIND_VIDEO && g_meeting_controller) {
      g_meeting_controller->enqueue(
          {MeetingController::CmdType::TrackPublishedDefend, identity, sid});
    }
  }

  // Phase 8.4.2: SDK 推 active speaker 列表过来；首个非本地 speaker 视为
  // 当前 active，投给 controller 做切换。MVP 不带 hold-down，看到抖动严重
  // 再加 600ms 阻尼。
  void onActiveSpeakersChanged(
      Room &, const ActiveSpeakersChangedEvent &ev) override {
    if (!g_meeting_controller)
      return;
    std::string new_active;
    for (auto *p : ev.speakers) {
      if (!p)
        continue;
      const std::string id = p->identity();
      if (id == local_identity_)
        continue; // 跳过自己
      new_active = id;
      break;
    }
    if (new_active.empty())
      return; // 全是本地 / 列表空，不切
    g_meeting_controller->enqueue(
        {MeetingController::CmdType::ActiveSpeakerObserved, new_active, ""});
  }

  void setLocalIdentity(std::string id) { local_identity_ = std::move(id); }

  void onTrackSubscribed(Room &room,
                         const TrackSubscribedEvent &ev) override {
    if (!ev.participant || !ev.track)
      return;
    const std::string identity = ev.participant->identity();
    const std::string track_name = ev.track->name();
    const TrackKind kind = ev.track->kind();

    const char *kind_str = (kind == TrackKind::KIND_VIDEO)   ? "video"
                           : (kind == TrackKind::KIND_AUDIO) ? "audio"
                                                              : "unknown";
    std::cout << "[loopback] track subscribed: " << identity << ":"
              << track_name << " kind=" << kind_str
              << " sid=" << ev.track->sid() << "\n";

    auto stats = g_stats.get(identity, track_name);
    if (kind == TrackKind::KIND_VIDEO) {
      // Phase 6.1.5 step B: 全链路 NV12 不再来回转 I420。
      // - 我们 MPP 解码器输出 NV12Buffer
      // - SDK FFI 的 cvt_nv12 路径（livekit-ffi/server/colorcvt）NV12→NV12
      //   走单次 nv12_copy memcpy
      // - 这里收到 NV12 直接喂 DRM plane（DRM_FORMAT_NV12），中间零格式转换
      // 之前 I420 路径每帧 720p 多吃 ~4 MB 内存带宽（解码侧 NV12→I420 +
      // 显示侧 I420→NV12），30fps 累计 ~120 MB/s，省下来 CPU 留给别的。
      VideoStream::Options vopts;
      vopts.format = VideoBufferType::NV12;
      vopts.capacity = 8; // ring buffer 防止积压
      room.setOnVideoFrameCallback(
          identity, track_name,
          [stats, identity](const VideoFrame &frame, std::int64_t /*ts*/) {
            stats->video_frames.fetch_add(1, std::memory_order_relaxed);
            if (!g_drm_ok.load(std::memory_order_relaxed) ||
                frame.type() != VideoBufferType::NV12)
              return;
            // Phase 8.4.1: 多 peer 走 router 网关，仅 active peer 进入
            // DrmDisplay。NV12 contiguous: data() = Y (w*h) + UV (w*h/2)。
            g_video_router.renderIfActive(identity, frame.data(),
                                          frame.width(), frame.height());
          },
          vopts);
    } else if (kind == TrackKind::KIND_AUDIO) {
      // 远端音频 track 的 name 经常是空字符串（比如来自手机端 WebRTC 的
      // 默认麦），by-name 注册不命中。改用 TrackSource enum：远端麦统一
      // 报告为 SOURCE_MICROPHONE。
      const TrackSource src = ev.track->source().value_or(
          TrackSource::SOURCE_MICROPHONE);
      room.setOnAudioFrameCallback(
          identity, src, [stats, identity](const AudioFrame &frame) {
            const auto n = stats->audio_frames.fetch_add(
                1, std::memory_order_relaxed);
            // 第一帧打印一次实际 PCM 参数，帮助排查音质问题
            if (n == 0) {
              std::cout << "[loopback] first audio frame [" << identity
                        << "]: rate=" << frame.sample_rate()
                        << "Hz channels=" << frame.num_channels()
                        << " samples_per_ch=" << frame.samples_per_channel()
                        << " buf_size=" << frame.data().size() << "\n";
            }
            // Phase 8.4.3: 不再直接 enqueue ALSA / processReverseStream。
            // 写到 g_mixer 的 per-peer ring，ALSA writer 线程按 10ms 节奏
            // pull 混音输出 → 单源馈给 ALSA + AEC reverse stream。
            const auto &samples = frame.data();
            if (!samples.empty()) {
              g_mixer.write(identity, samples.data(),
                            frame.samples_per_channel());
            }
          });
    }
  }

private:
  std::string local_identity_; // Phase 8.4.2: 给 onActiveSpeakersChanged 过滤
};

// Phase 8.4.4-完整: per-meeting FFI + APM 初始化。每次会议开始前调用，
// 会议结束 livekit::shutdown 之后再下次会议会重新初始化。
//
// 为什么每会议重做：Room 析构是 fire-and-forget drop FFI handle，server 收到
// disconnect 是异步 + 不保证下个会议开始前完成；上轮残余 RTP / subscription
// thread / 帧 callback 会泄漏到下轮，导致手机端仍看到板子在旧会议中。完整
// shutdown FFI 强制同步关连接 + 清子线程。代价：每会议多花 ~100-300ms 启动。
static void setupFfiAndApm() {
  livekit::initialize(LogLevel::Info, LogSink::kConsole);

  // Phase 7.4 APM tuning 见 phase7-summary.md / phase8-summary.md 8.1
  if (const char *e = std::getenv("BOARD_LOOPBACK_AEC");
      e && std::string(e) == "1") {
    AudioProcessingModule::Options opts;
    opts.echo_cancellation = true;
    opts.noise_suppression = true;
    opts.high_pass_filter = true;
    // AGC 继续关 — ES8389 硬件 PGA + 8× 软件 gain 已校准
    opts.auto_gain_control = false;
    g_apm = std::make_unique<AudioProcessingModule>(opts);
    g_apm_delay_ms = 400;
    if (const char *d = std::getenv("BOARD_LOOPBACK_AEC_DELAY_MS")) {
      try {
        g_apm_delay_ms = std::stoi(d);
      } catch (...) { /* keep default */
      }
    }
    g_apm->setStreamDelayMs(g_apm_delay_ms);
    g_apm_ready.store(true);
    std::cout << "[loopback] APM AEC enabled (echo_cancellation + "
                 "noise_suppression + high_pass_filter), delay_ms="
              << g_apm_delay_ms << "\n";
  }
}

// 拆 FFI + APM。在 room.reset() 之后调，下次会议前 setupFfiAndApm 重建。
static void teardownFfiAndApm() {
  g_apm.reset();
  g_apm_ready.store(false);
  livekit::shutdown();
}

// Phase 8.4.4-完整 debug: SIGSEGV/SIGBUS/SIGABRT backtrace handler。
// active speaker 切换时偶发 segfault，trace 输出到 stderr 让 smoke.sh
// 抓得到。仅用 async-signal-safe 函数（backtrace + backtrace_symbols_fd
// 安全），不调 malloc / printf / std::cout。
static void crashHandler(int sig) {
  static const char banner[] =
      "\n=== BoardLoopback CRASH backtrace ===\nsignal=";
  ::write(STDERR_FILENO, banner, sizeof(banner) - 1);
  char sigbuf[16] = {0};
  int n = sig, len = 0;
  if (n < 0) { sigbuf[len++] = '-'; n = -n; }
  char rev[10]; int rl = 0;
  if (n == 0) rev[rl++] = '0';
  while (n) { rev[rl++] = char('0' + n % 10); n /= 10; }
  while (rl) sigbuf[len++] = rev[--rl];
  sigbuf[len++] = '\n';
  ::write(STDERR_FILENO, sigbuf, len);

  void *frames[64];
  int nframes = backtrace(frames, 64);
  backtrace_symbols_fd(frames, nframes, STDERR_FILENO);
  static const char tail[] = "=== end backtrace ===\n";
  ::write(STDERR_FILENO, tail, sizeof(tail) - 1);

  // 恢复默认 handler 重新触发 → 让 kernel 写 core file
  std::signal(sig, SIG_DFL);
  ::raise(sig);
}

int main(int argc, char *argv[]) {
  std::signal(SIGINT, handleSignal);
#ifdef SIGTERM
  std::signal(SIGTERM, handleSignal);
#endif
  std::signal(SIGSEGV, crashHandler);
  std::signal(SIGBUS, crashHandler);
  std::signal(SIGABRT, crashHandler);

  // Phase 8.4.4: BOARD_API_TOKEN 是 HTTP API 的 Bearer，必填（生产防同
  // LAN 误拽）。MVP 不用 dev-mode 后门，未设直接退。
  std::string api_token = getenvOrEmpty("BOARD_API_TOKEN");
  if (api_token.empty()) {
    std::cerr << "ERROR: BOARD_API_TOKEN env not set; required for HTTP API "
                 "Bearer auth. Set a non-empty secret and retry.\n";
    return 1;
  }
  int api_port = 8080;
  if (const char *p = std::getenv("BOARD_API_PORT")) {
    try {
      api_port = std::stoi(p);
    } catch (...) { /* keep default */ }
  }

  // Phase 8.4.4-完整: livekit::initialize + APM 创建移到 per-meeting 生命
  // 周期里（每会议 init + shutdown），保证 server side 真收到 disconnect
  // 信号——否则 Room 析构只是 fire-and-forget drop FFI handle，会议结束
  // 后手机/server 仍认为板子在房间，下次 join 同 identity 会被 server 误
  // 判为 reconnect。详见下面外循环里的 setupFfiAndApm() 调用。
  //
  // APM tuning: BOARD_LOOPBACK_AEC=1 启用，BOARD_LOOPBACK_AEC_DELAY_MS 调
  // stream delay。ATK-DLRV1126B PulseAudio→ES8389 实测最佳 400ms（详见
  // Phase 7.4 / 8.1 sweep 结果，smoke.sh 默认 300ms）。

  // Try to take over the DRM device for direct video plane rendering.
  // Skip with BOARD_LOOPBACK_NO_DRM=1 if running headless or weston
  // can't be stopped.
  if (!std::getenv("BOARD_LOOPBACK_NO_DRM")) {
    if (g_drm.open()) {
      g_drm_ok.store(true);
      std::cout << "[loopback] DRM display ready — incoming video will "
                   "render to MIPI panel\n";
    } else {
      std::cerr << "[loopback] DRM init failed; continuing without on-screen "
                   "render. Stop weston (e.g. `pkill weston`) and retry, "
                   "or set BOARD_LOOPBACK_NO_DRM=1 to silence.\n";
    }
  }

  // Phase 8.4.4: 起 HTTP API server，等 /v1/meeting/join 把 (url, token)
  // 投到 MeetingState 后再继续。如果启动时已有 LIVEKIT_URL/LIVEKIT_TOKEN
  // env 或 argv，内部 fire 一次 join request 实现"自动加入"（兼容老的
  // smoke.sh + 测试脚本，不强迫用户必须用 curl 才能开会）。
  BoardHttpServer http_server(api_token, api_port);
  http_server.start();

  // Phase 8.4.4-完整: daemon 外循环。每轮：
  //   1. waitForJoinOrShutdown 阻塞等下一次 /join 或 SIGINT/SIGTERM
  //   2. 收到 join → 跑会议体（在 do/while(false) 块作用域内，所有
  //      per-meeting 资源声明在块内 → 块退栈时 RAII 清理）
  //   3. clearMeeting 回 Idle，外循环继续
  // 启动时 LIVEKIT_URL/TOKEN env 已设的话，先 fire 第一次 join request
  // 实现"自动加入"（兼容老 smoke.sh / 测试脚本 / 一会议一进程模式）。
  {
    std::string env_url = getenvOrEmpty("LIVEKIT_URL");
    std::string env_token = getenvOrEmpty("LIVEKIT_TOKEN");
    if (argc >= 3) {
      env_url = argv[1];
      env_token = argv[2];
    }
    if (!env_url.empty() && !env_token.empty()) {
      std::cout << "[loopback] LIVEKIT_URL/TOKEN provided, auto-firing first "
                   "join (skip HTTP wait)\n";
      g_meeting_state.requestJoin(env_url, env_token);
    }
  }

  // Phase 8.4.5: pre-compute V4L2 device + resolution for the idle preview
  // thread. These values come from env vars that don't change after startup.
  const bool preview_synth = (std::getenv("BOARD_LOOPBACK_SYNTH_VIDEO") != nullptr);
  const std::string preview_v4l2_dev = []{
    const char *e = std::getenv("V4L2_DEVICE");
    return (e && *e) ? std::string(e) : std::string("/dev/video-camera0");
  }();
  VideoResolution preview_res = kDefaultRes;
  if (const char *e = std::getenv("BOARD_LOOPBACK_VIDEO_RES")) {
    std::string s(e);
    for (auto &c : s) c = static_cast<char>(std::tolower(c));
    if      (s == "sd")  preview_res = kResSD;
    else if (s == "hd")  preview_res = kResHD;
    else if (s == "fhd") preview_res = kResFHD;
  }

  while (true) {
    if (g_meeting_state.isShutdown())
      break;

    // Phase 8.4.5: idle preview — capture local camera and show on screen
    // while waiting for next /join. The thread stops and releases V4L2 before
    // the meeting body opens its own V4l2Capture instance.
    std::atomic<bool> preview_stop{false};
    std::thread preview_thread;
    if (!preview_synth && g_drm_ok.load()) {
      preview_thread = std::thread(
          [&preview_stop, preview_v4l2_dev, preview_res]() {
            V4l2Capture pv;
            if (!pv.open(preview_v4l2_dev.c_str(), preview_res.width,
                         preview_res.height, preview_res.fps) ||
                !pv.startStream())
              return;
            std::vector<std::uint8_t> bytes;
            while (!preview_stop.load(std::memory_order_relaxed)) {
              if (pv.dequeue(bytes, 200))
                g_drm.renderNV12Rotated(bytes.data(), pv.width(), pv.height());
            }
          });
    }

    std::cout << "[daemon] waiting for HTTP /v1/meeting/join "
                 "(POST to 0.0.0.0:"
              << api_port << " with Bearer auth) or SIGINT/SIGTERM...\n";
    auto join_or = g_meeting_state.waitForJoinOrShutdown();

    // Stop idle preview before the meeting body opens V4L2.
    preview_stop.store(true);
    if (preview_thread.joinable()) preview_thread.join();

    if (!join_or) {
      std::cout << "[daemon] shutdown requested, exiting outer loop\n";
      break;
    }
    auto [url, token] = *join_or;
    std::cout << "[loopback] join received, connecting to '" << url << "'\n";
    g_running.store(true);

  // Phase 8.4.4-完整: 每会议 init FFI + APM（保证上轮残余完全 tear down）
  setupFfiAndApm();

  // === 会议体起点：以下所有 per-meeting 变量都在外循环这一轮的栈帧上 ===
  auto room = std::make_unique<Room>();
  LoopbackDelegate delegate;
  room->setDelegate(&delegate);

  RoomOptions options;
  options.auto_subscribe = true;
  options.dynacast = false;
  if (!room->Connect(url, token, options)) {
    std::cerr << "[loopback] failed to connect, returning to idle\n";
    room.reset();
    teardownFfiAndApm(); // 失败也要拆 FFI，下次 setupFfiAndApm 才有干净状态
    g_in_meeting.store(false); // Phase 8.4.5: back to preview mode
    g_meeting_state.clearMeeting();
    continue; // 不退 daemon，回外循环等下次 /join
  }

  LocalParticipant *lp = room->localParticipant();
  std::cout << "[loopback] connected as identity='" << lp->identity()
            << "' room='" << room->room_info().name << "'\n";

  // Phase 8.4.4: 状态切到 InMeeting，让 GET /status 能 report room_name +
  // 通过 Room 指针拿 participant_count。
  g_meeting_state.setInMeeting(room->room_info().name, room.get());
  g_in_meeting.store(true); // Phase 8.4.5: switch VideoRouter to split-screen

  // Phase 8.4.2: 起 MeetingController，将 SDK delegate 投来的 active
  // speaker / track published 事件串行 reconcile（避免多 delegate 线程
  // 并发改 setSubscribed / VideoRouter active 状态）。delegate 的
  // local_identity 也得知道，以便 onActiveSpeakersChanged 跳过自己。
  delegate.setLocalIdentity(lp->identity());
  g_meeting_controller =
      std::make_unique<MeetingController>(room.get(), lp->identity());

  // Phase 8.4.3: ALSA playback 提前开（不再 lazy 等第一帧），让 pull 模型
  // 的 writer 线程一启动就有 pcm_ 可用。所有远端 audio 已知 48kHz mono
  // (LiveKit FFI 输出固定，跟 AudioMixer::kSampleRate / kChannels 对齐)。
  if (!g_alsa.ensureOpen(AudioMixer::kSampleRate, AudioMixer::kChannels)) {
    std::cerr << "[loopback] WARN: ALSA playback open failed at startup; "
                 "remote audio will be silent\n";
  }

  // Publish video. Try opening the on-board V4L2 camera (rkisp_mainpath
  // via /dev/video-camera0); fall back to synthetic gradient if it fails
  // or BOARD_LOOPBACK_SYNTH_VIDEO=1 is set.
  const bool synth_video =
      std::getenv("BOARD_LOOPBACK_SYNTH_VIDEO") != nullptr;
  // Resolution preset selection: BOARD_LOOPBACK_VIDEO_RES = sd | hd | fhd
  VideoResolution selected_res = kDefaultRes;
  if (const char *res_env = std::getenv("BOARD_LOOPBACK_VIDEO_RES")) {
    std::string s(res_env);
    for (auto &c : s) c = static_cast<char>(std::tolower(c));
    if      (s == "sd")  selected_res = kResSD;
    else if (s == "hd")  selected_res = kResHD;
    else if (s == "fhd") selected_res = kResFHD;
    else std::cerr << "[loopback] unknown BOARD_LOOPBACK_VIDEO_RES='"
                   << res_env << "', using default " << kDefaultRes.name
                   << "\n";
  }
  std::cout << "[loopback] requested resolution: " << selected_res.name
            << " (" << selected_res.width << "x" << selected_res.height
            << "@" << selected_res.fps << "fps)\n";
  // Note: rotation is consumed by the MPP encoder via env, not by the
  // VideoSource::captureFrame rotation parameter. RTP rotation extension
  // didn't actually take effect with the current receiver; the encoder
  // pre-rotates the bitstream and ships it upright on the wire instead.

  std::unique_ptr<V4l2Capture> v4l2;
  int video_w = selected_res.width, video_h = selected_res.height;
  if (!synth_video) {
    const char *v4l2_dev_env = std::getenv("V4L2_DEVICE");
    const char *v4l2_dev =
        (v4l2_dev_env && *v4l2_dev_env) ? v4l2_dev_env : "/dev/video-camera0";
    v4l2 = std::make_unique<V4l2Capture>();
    if (v4l2->open(v4l2_dev, selected_res.width, selected_res.height,
                   selected_res.fps) &&
        v4l2->startStream()) {
      video_w = v4l2->width();
      video_h = v4l2->height();
    } else {
      std::cerr << "[loopback] v4l2 init failed; falling back to synth gradient\n";
      v4l2.reset();
    }
  }
  auto video_source = std::make_shared<VideoSource>(video_w, video_h);
  // Phase 6 prep: 显式声明 H.264 编码（默认 SDP 协商优先 VP8 因为 livekit-ffi
  // 的 codec template 顺序 VP8 → H.264 → AV1 → VP9，发布端要主动指定 H.264 才能
  // 跟 RV1126B MPP 编码器匹配。BOARD_LOOPBACK_VIDEO_CODEC=vp8/h264/h265 可调。
  livekit::VideoCodec vc = livekit::VideoCodec::H264;
  if (const char *vc_env = std::getenv("BOARD_LOOPBACK_VIDEO_CODEC")) {
    std::string s(vc_env);
    if (s == "vp8") vc = livekit::VideoCodec::VP8;
    else if (s == "vp9") vc = livekit::VideoCodec::VP9;
    else if (s == "h265") vc = livekit::VideoCodec::H265;
    else if (s == "av1") vc = livekit::VideoCodec::AV1;
    // h264 is default
  }
  const char *vc_name = (vc == livekit::VideoCodec::H264) ? "H264"
                      : (vc == livekit::VideoCodec::VP8)  ? "VP8"
                      : (vc == livekit::VideoCodec::VP9)  ? "VP9"
                      : (vc == livekit::VideoCodec::H265) ? "H265"
                      :                                    "AV1";
  auto video_track = LocalVideoTrack::createLocalVideoTrack(
      kVideoTrackName, video_source);
  TrackPublishOptions vopts_pub;
  vopts_pub.source = TrackSource::SOURCE_CAMERA;
  vopts_pub.video_codec = vc;
  // Single-publisher single-resolution setup: disable simulcast (no need
  // to triple-encode for diverse subscribers) and pin max_bitrate so the
  // top layer actually gets the bandwidth it needs. Without this, BWE's
  // default 3-layer split gives 720p only ~30 kbps and the picture is a
  // block-heavy mush.
  vopts_pub.simulcast = false;
  VideoEncodingOptions venc;
  venc.max_bitrate = selected_res.max_bitrate_bps;
  venc.max_framerate = static_cast<double>(selected_res.fps);
  vopts_pub.video_encoding = venc;
  lp->publishTrack(video_track, vopts_pub);
  std::cout << "[loopback] publishing video '" << kVideoTrackName << "' "
            << video_w << "x" << video_h << " codec=" << vc_name << " ("
            << (v4l2 ? "live camera NV12" : "synth gradient RGBA") << ")\n";

  // Publish audio from the board's real microphone (ES8389 codec capture
  // via ALSA). Falls back to silence if capture device can't open. Set
  // BOARD_LOOPBACK_SYNTH_AUDIO=1 to override and use a 440Hz sine instead
  // (useful for environment that has no mic or for repro the older
  // synthetic-only mode).
  const bool synth_audio =
      std::getenv("BOARD_LOOPBACK_SYNTH_AUDIO") != nullptr;
  auto audio_source = std::make_shared<AudioSource>(kAudioSampleRate,
                                                    kAudioChannels);
  std::shared_ptr<LocalAudioTrack> audio_track = lp->publishAudioTrack(
      kAudioTrackName, audio_source, TrackSource::SOURCE_MICROPHONE);

  // Try opening ALSA capture (mic). If that succeeds, the audio_thread
  // below pumps real mic samples; otherwise it falls back to synthetic
  // (or refuses if synth_audio mode is off and capture failed).
  snd_pcm_t *capture_pcm = nullptr;
  if (!synth_audio) {
    const char *cap_dev_env = std::getenv("ALSA_CAPTURE_DEVICE");
    const char *cap_dev =
        (cap_dev_env && *cap_dev_env) ? cap_dev_env : "default";
    int err = snd_pcm_open(&capture_pcm, cap_dev, SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
      std::cerr << "[loopback] alsa capture open(" << cap_dev
                << ") failed: " << snd_strerror(err)
                << " — falling back to synthetic 440Hz tone\n";
      capture_pcm = nullptr;
    } else {
      err = snd_pcm_set_params(capture_pcm, SND_PCM_FORMAT_S16_LE,
                               SND_PCM_ACCESS_RW_INTERLEAVED,
                               kAudioChannels, kAudioSampleRate,
                               /*soft_resample=*/1,
                               /*latency_us=*/200000); // 200ms 录音 buffer
      if (err < 0) {
        std::cerr << "[loopback] alsa capture set_params failed: "
                  << snd_strerror(err) << " — falling back to sine\n";
        snd_pcm_close(capture_pcm);
        capture_pcm = nullptr;
      } else {
        std::cout << "[loopback] alsa capture opened: " << cap_dev
                  << " " << kAudioSampleRate << "Hz " << kAudioChannels
                  << "ch s16le 200ms-buf\n";
      }
    }
  }
  std::cout << "[loopback] publishing audio '" << kAudioTrackName << "' "
            << kAudioSampleRate << "Hz mono "
            << (capture_pcm ? "live mic" : "synth 440Hz sine") << "\n";

  // ES8389 mic 默认增益偏低，先用纯软件 gain 提一下。BOARD_LOOPBACK_MIC_GAIN
  // 可改。1.0 关闭软件放大走原始电平。
  //
  // 历史：早先默认 12× 是因为 SDK 内部的 AGC 又把信号压回去（webrtc PCF 默认
  // 装 BuiltinAudioProcessingBuilder 开启 AGC1 + AGC2），所以表面上 12× 实际
  // 远端听到的还是软。webrtc-sys/peer_connection_factory.cpp 里关掉 AGC 后，
  // 12× 会真的 12×，几乎肯定 clip 失真。8× 是安全 headroom；如果板上 mic 装
  // 配后噪音底大或离嘴远，env 调到 10-12；如果离嘴近，env 调到 4-6 避失真。
  float mic_gain = 8.0f;
  if (const char *g_env = std::getenv("BOARD_LOOPBACK_MIC_GAIN")) {
    try {
      mic_gain = std::stof(g_env);
    } catch (...) {
      // ignore parse error, keep default
    }
  }
  if (capture_pcm) {
    std::cout << "[loopback] mic software gain = " << mic_gain << "×\n";
  }

  std::thread audio_thread([&, capture_pcm]() {
    std::vector<std::int16_t> buf(kAudioSamplesPerChannel * kAudioChannels);
    double phase = 0.0;
    const double phase_step = 2.0 * 3.14159265358979323846 *
                              kAudioToneHz / kAudioSampleRate;
    while (g_running.load()) {
      if (capture_pcm) {
        // Real mic path: blocking read of 10ms (480) frames
        snd_pcm_sframes_t got = snd_pcm_readi(
            capture_pcm, buf.data(), kAudioSamplesPerChannel);
        if (got < 0) {
          const int e = static_cast<int>(got);
          got = snd_pcm_recover(capture_pcm, e, /*silent=*/1);
          if (got < 0) {
            std::cerr << "[loopback] alsa capture err: "
                      << snd_strerror(static_cast<int>(got)) << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
          }
          // recovered — treat as a single dropped frame, continue
          continue;
        }
        if (got < kAudioSamplesPerChannel) {
          // Short read — pad rest with zeros
          for (size_t i = got * kAudioChannels; i < buf.size(); ++i)
            buf[i] = 0;
        }
        // Apply software gain with hard clipping at int16 range
        if (mic_gain != 1.0f) {
          for (auto &s : buf) {
            const float v = static_cast<float>(s) * mic_gain;
            s = static_cast<std::int16_t>(
                std::max(-32768.0f, std::min(32767.0f, v)));
          }
        }
      } else {
        // Fallback synthetic 440Hz
        for (int i = 0; i < kAudioSamplesPerChannel; ++i) {
          buf[i] = static_cast<std::int16_t>(
              kAudioAmplitude * std::sin(phase));
          phase += phase_step;
          if (phase > 2.0 * 3.14159265358979323846)
            phase -= 2.0 * 3.14159265358979323846;
        }
      }

      AudioFrame frame(buf, kAudioSampleRate, kAudioChannels,
                       kAudioSamplesPerChannel);
      // Phase 7.4: run the captured mic frame through APM. AEC subtracts
      // the speaker reference signal it learned from processReverseStream,
      // NS removes background noise, HPF cuts <80Hz rumble — all in-place.
      // The processed buffer is what we publish to remote.
      if (g_apm_ready.load()) {
        try {
          g_apm->processStream(frame);
        } catch (const std::exception &e) {
          std::cerr << "[loopback] APM processStream err: " << e.what() << "\n";
        }
      }
      try {
        audio_source->captureFrame(frame, /*timeout_ms=*/100);
      } catch (const std::exception &e) {
        std::cerr << "[loopback] audio capture push err: " << e.what()
                  << "\n";
      }

      if (!capture_pcm) {
        // Synth mode — pace ourselves; mic mode is naturally paced by
        // snd_pcm_readi blocking on hardware.
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kAudioFrameMs));
      }
    }

    if (capture_pcm) {
      snd_pcm_drop(capture_pcm);
      snd_pcm_close(capture_pcm);
    }
  });

  const auto t0 = std::chrono::steady_clock::now();
  auto last_report = t0;
  std::uint64_t frame_count = 0;

  std::cout << "[loopback] running until Ctrl-C; reporting every "
            << kReportEverySeconds << "s.\n";

  while (g_running.load()) {
    if (v4l2) {
      // Real V4L2 capture — pull next NV12 frame from rkisp_mainpath
      std::vector<std::uint8_t> bytes;
      if (v4l2->dequeue(bytes, /*timeout_ms=*/200)) {
        // Phase 8.4.5: stash a copy for the local half of split-screen
        {
          std::lock_guard<std::mutex> lk(g_local_frame_mu);
          g_local_frame_data = bytes;
          g_local_frame_w    = video_w;
          g_local_frame_h    = video_h;
        }
        // Phase 8.4.5: show local camera preview when not in meeting OR
        // when in meeting but no remote peer is active (peer disconnected).
        const bool no_remote = g_video_router.getActive().empty();
        if ((!g_in_meeting.load(std::memory_order_relaxed) || no_remote)
            && g_drm_ok.load())
          g_drm.renderNV12Rotated(bytes.data(), video_w, video_h);

        VideoFrame vf(video_w, video_h, VideoBufferType::NV12,
                      std::move(bytes));
        video_source->captureFrame(std::move(vf));
      } else {
        // Timeout — camera may be slow to bring up; just skip this iteration
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
        continue;
      }
    } else {
      // Synthetic fallback: time-modulated RGBA gradient
      VideoFrame vf = VideoFrame::create(video_w, video_h,
                                         VideoBufferType::RGBA);
      const auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      const std::uint8_t hue = static_cast<std::uint8_t>((t_ms / 30) & 0xff);
      std::uint8_t *p = vf.data();
      if (p) {
        for (int y = 0; y < video_h; ++y) {
          for (int x = 0; x < video_w; ++x) {
            p[0] = static_cast<std::uint8_t>(x * 255 / video_w);  // R
            p[1] = static_cast<std::uint8_t>(y * 255 / video_h);  // G
            p[2] = hue;                                            // B
            p[3] = 255;                                            // A
            p += 4;
          }
        }
      }
      video_source->captureFrame(std::move(vf));
    }

    // 注：data track publish 在当前 server/NAT 路径下 SCTP 协商超时，
    // 已从该 example 移除以避免每次都报错（见 phase5-summary §polish 项）。
    // 视频/音频是主轨道走 RTP/SRTP，不受 SCTP 协商失败影响。

    // Periodic report
    const auto now = std::chrono::steady_clock::now();
    if (now - last_report >= std::chrono::seconds(kReportEverySeconds)) {
      last_report = now;
      const auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                            now - t0)
                            .count();
      std::cout << "[loopback] T+" << secs << "s  published " << frame_count
                << " video frames; alsa_underruns="
                << g_alsa.underruns()
                << " mixer_dropped=" << g_mixer.dropped() << "\n";
      std::cout << "[loopback] remote streams:\n";
      g_stats.print(std::cout);
    }

    ++frame_count;
    // V4L2 dequeue blocks on poll → naturally paced by camera fps;
    // synth path needs explicit pacing.
    if (!v4l2)
      std::this_thread::sleep_for(std::chrono::milliseconds(33));
  }

  std::cout << "[loopback] final stats:\n";
  g_stats.print(std::cout);

  std::cout << "[loopback] disconnecting...\n";
  if (audio_thread.joinable())
    audio_thread.join();
  // Phase 8.4.2: 先停 controller worker（避免 worker 在 room 已 reset 后
  // 还尝试访问 remoteParticipant）。它 dtor 会发 Shutdown cmd + join。
  g_meeting_controller.reset();
  // Phase 8.4.4: 先清掉 MeetingState 的 Room 指针（否则 status endpoint
  // 拿到悬垂指针），然后再 reset Room。
  g_in_meeting.store(false); // Phase 8.4.5: back to local preview mode
  g_meeting_state.clearMeeting();
  // Drop tracks first
  audio_track.reset();
  audio_source.reset();
  video_track.reset();
  // Phase 8.4.4-完整: 显式断开 Room（送 DisconnectRequest 到 FFI）
  // 让 server 立刻 LEAVE 房间，对端立刻看到我们离开 —— 不再等 ~30s
  // 客户端超时。Room::Disconnect 阻塞等 FFI 回调，正常 < 200ms。
  if (room) {
    room->Disconnect();
  }
  room.reset();
  // 然后 tear-down FFI 整个 runtime，下次 setupFfiAndApm 重建。
  teardownFfiAndApm();
  // === 会议体终点：所有 per-meeting 资源 RAII + FFI 都拆完了 ===
  std::cout << "[daemon] meeting cleaned up; idle, waiting for next /join "
               "or signal\n";
  } // ← 外循环 while(true) 的尾巴

  // Phase 8.4.4-完整: 进程级清理（外循环退出后才走到这里）。
  // 注意：此时 livekit FFI 已经被上一轮 teardownFfiAndApm 清掉了（如果
  // 跑过至少一次会议），所以这里不再 livekit::shutdown。HTTP server 还
  // 活着，最后停掉。
  std::cout << "[daemon] outer loop exited, shutting down process-level "
               "resources\n";
  http_server.stop();

  std::cout << "[loopback] done.\n";
  return 0;
}
