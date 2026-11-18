#include "dsp.hpp"
#include "threaded_inference.hpp"
#include "model.hpp"
#include "tensor.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <libnyquist/Common.h>
#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>
#include <sstream>
#include <string>
#include <unsupported/Eigen/FFT>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>

using namespace demucscpp;
using namespace nqr;

static Eigen::MatrixXf load_audio_file(std::string filename)
{
    // load a wav file with libnyquist
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();

    NyquistIO loader;

    loader.Load(fileData.get(), filename);

    if (fileData->sampleRate != demucscpp::SUPPORTED_SAMPLE_RATE)
    {
        std::cerr << "[ERROR] demucs.cpp only supports the following sample "
                     "rate (Hz): "
                  << SUPPORTED_SAMPLE_RATE << std::endl;
        exit(1);
    }

    std::cout << "Input samples: "
              << fileData->samples.size() / fileData->channelCount << std::endl;
    std::cout << "Length in seconds: " << fileData->lengthSeconds << std::endl;
    std::cout << "Number of channels: " << fileData->channelCount << std::endl;

    if (fileData->channelCount != 2 && fileData->channelCount != 1)
    {
        std::cerr << "[ERROR] demucs.cpp only supports mono and stereo audio"
                  << std::endl;
        exit(1);
    }

    // number of samples per channel
    size_t N = fileData->samples.size() / fileData->channelCount;

    // create a struct to hold two float vectors for left and right channels
    Eigen::MatrixXf ret(2, N);

    if (fileData->channelCount == 1)
    {
        // Mono case
        for (size_t i = 0; i < N; ++i)
        {
            ret(0, i) = fileData->samples[i]; // left channel
            ret(1, i) = fileData->samples[i]; // right channel
        }
    }
    else
    {
        // Stereo case
        for (size_t i = 0; i < N; ++i)
        {
            ret(0, i) = fileData->samples[2 * i];     // left channel
            ret(1, i) = fileData->samples[2 * i + 1]; // right channel
        }
    }

    return ret;
}

// write a function to write a StereoWaveform to a wav file
static void write_audio_file(const Eigen::MatrixXf &waveform,
                             std::string filename)
{
    // create a struct to hold the audio data
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();

    // set the sample rate
    fileData->sampleRate = SUPPORTED_SAMPLE_RATE;

    // set the number of channels
    fileData->channelCount = 2;

    // set the number of samples
    fileData->samples.resize(waveform.cols() * 2);

    // write the left channel
    for (long int i = 0; i < waveform.cols(); ++i)
    {
        fileData->samples[2 * i] = waveform(0, i);
        fileData->samples[2 * i + 1] = waveform(1, i);
    }

    int encoderStatus =
        encode_wav_to_disk({fileData->channelCount, PCM_FLT, DITHER_TRIANGLE},
                           fileData.get(), filename);
    std::cout << "Encoder Status: " << encoderStatus << std::endl;
}

#ifdef DEMUCS_OPENBLAS_SET_THREADS
extern "C" void openblas_set_num_threads(int num_threads);
#endif

// Engelmann (MovieJack): Treiber mit automatischem Multithreading und
// maschinenlesbarem Fortschritt.
//
//   demucs-cpp <model> <wav> <out dir> [threads]
//
// Ohne [threads] wird automatisch gewaehlt (halbe logische Kerne, max. 4,
// mind. ~15 s Audio pro Thread). Override auch per Umgebung MJ_DEMUCS_THREADS.
// Fortschritt: Zeilen "MJ_PROGRESS <0..100>" auf stdout (gesamt ueber alle
// Threads), dazu "MJ_THREADS <n>". Der Aufruf mit drei Argumenten bleibt
// kompatibel zu aelteren MovieJack-Versionen.
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

// Freier Arbeitsspeicher in MB (0 = unbekannt). Jeder Thread braucht
// gemessen gut 2 GB — mehr Threads als RAM-Platz waeren langsamer statt
// schneller (Auslagerung) oder enden im Speicherabbruch.
static double available_mb()
{
#if defined(_WIN32)
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st))
        return (double)st.ullAvailPhys / (1024.0 * 1024.0);
    return 0.0;
#elif defined(__APPLE__)
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm, &count) == KERN_SUCCESS)
    {
        double page = (double)vm_kernel_page_size;
        return ((double)vm.free_count + (double)vm.inactive_count) * page /
               (1024.0 * 1024.0);
    }
    return 0.0;
#else
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page > 0)
        return (double)pages * (double)page / (1024.0 * 1024.0);
    return 0.0;
#endif
}

static int pick_threads(int argc, const char **argv, double seconds)
{
    int n = 0;
    if (argc >= 5)
        n = std::atoi(argv[4]);
    if (n <= 0)
    {
        const char *env = std::getenv("MJ_DEMUCS_THREADS");
        if (env)
            n = std::atoi(env);
    }
    if (n <= 0)
    {
        unsigned hw = std::thread::hardware_concurrency();
        n = hw >= 4 ? (int)(hw / 2) : 1;
        n = std::min(n, 4);
    }
    double mb = available_mb();
    if (mb > 0.0)
    {
        int by_ram = (int)((mb - 512.0) / 2560.0);
        n = std::min(n, std::max(1, by_ram));
    }
    int by_len = (int)(seconds / 15.0);
    n = std::max(1, std::min(n, std::max(1, by_len)));
    return std::min(n, 16);
}

int main(int argc, const char **argv)
{
    if (argc != 4 && argc != 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <model file> <wav file> <out dir> [num threads]"
                  << std::endl;
        exit(1);
    }

#ifdef DEMUCS_OPENBLAS_SET_THREADS
    // Wir parallelisieren selbst ueber Segmente — sonst ueberbuchen
    // zusaetzliche OpenBLAS-Threads die CPU.
    openblas_set_num_threads(1);
#endif

    std::cout << "demucs-cpp (Engelmann) driver program" << std::endl;

    std::string model_file = argv[1];
    std::string wav_file = argv[2];
    std::string out_dir = argv[3];

    Eigen::MatrixXf audio = load_audio_file(wav_file);
    Eigen::Tensor3dXf out_targets;

    struct demucs_model model
    {
    };

    auto ret = load_demucs_model(model_file, &model);
    std::cout << "demucs_model_load returned " << (ret ? "true" : "false")
              << std::endl;
    if (!ret)
    {
        std::cerr << "Error loading model" << std::endl;
        exit(1);
    }

    int nb_sources = model.is_4sources ? 4 : 6;
    double seconds =
        (double)audio.cols() / (double)demucscpp::SUPPORTED_SAMPLE_RATE;
    int num_threads = pick_threads(argc, argv, seconds);

    std::cout << "Starting Demucs (" << std::to_string(nb_sources)
              << "-source) inference" << std::endl;
    std::cout << "MJ_THREADS " << num_threads << " ram_free_mb "
              << (long)available_mb() << std::endl;

    std::mutex out_mtx;
    std::vector<float> per_thread(num_threads, 0.0f);
    float last_reported = -1.0f;
    std::function<void(int, float)> report = [&](int idx, float progress)
    {
        std::lock_guard<std::mutex> lock(out_mtx);
        if (idx >= 0 && idx < (int)per_thread.size())
            per_thread[idx] = std::max(per_thread[idx], progress);
        float sum = 0.0f;
        for (float v : per_thread)
            sum += v;
        float overall = 100.0f * sum / (float)per_thread.size();
        if (overall >= last_reported + 0.5f)
        {
            last_reported = overall;
            std::cout << "MJ_PROGRESS " << std::fixed << std::setprecision(1)
                      << overall << std::endl;
        }
    };

    Eigen::Tensor3dXf audio_targets;
    if (num_threads <= 1)
    {
        demucscpp::ProgressCallback cb =
            [&](float progress, const std::string &) { report(0, progress); };
        audio_targets = demucscpp::demucs_inference(model, audio, cb);
    }
    else
    {
        audio_targets = demucscppthreaded::threaded_inference(
            model, audio, num_threads, "", report);
    }
    std::cout << "MJ_PROGRESS 100.0" << std::endl;

    out_targets = audio_targets;

    int nb_out_sources = model.is_4sources ? 4 : 6;

    for (int target = 0; target < nb_out_sources; ++target)
    {
        // now write the 4 audio waveforms to files in the output dir
        // using libnyquist
        // join out_dir with "/target_0.wav"
        // using std::filesystem::path;

        std::filesystem::path p = out_dir;
        // make sure the directory exists
        std::filesystem::create_directories(p);

        auto p_target = p / "target_0.wav";

        // target 0,1,2,3 map to drums,bass,other,vocals

        std::string target_name;

        switch (target)
        {
        case 0:
            target_name = "drums";
            break;
        case 1:
            target_name = "bass";
            break;
        case 2:
            target_name = "other";
            break;
        case 3:
            target_name = "vocals";
            break;
        case 4:
            target_name = "guitar";
            break;
        case 5:
            target_name = "piano";
            break;
        default:
            std::cerr << "Error: target " << target << " not supported"
                      << std::endl;
            exit(1);
        }

        // insert target_name into the path after the digit
        // e.g. target_name_0_drums.wav
        p_target.replace_filename("target_" + std::to_string(target) + "_" +
                                  target_name + ".wav");

        std::cout << "Writing wav file " << p_target << std::endl;

        Eigen::MatrixXf target_waveform(2, audio.cols());

        // copy the input stereo wav file into all 4 targets
        for (int channel = 0; channel < 2; ++channel)
        {
            for (int sample = 0; sample < audio.cols(); ++sample)
            {
                target_waveform(channel, sample) =
                    out_targets(target, channel, sample);
            }
        }

        write_audio_file(target_waveform, p_target.string());
    }
}
