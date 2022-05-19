#pragma once
#include <chrono>
#include <string>
#include <iostream>

class Timer
{
private:
    float &_time_dst;
    bool _print;
    std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

public:
    Timer(float &time_dst, bool print)
    : _time_dst(time_dst), _print(print)
    {
        _start_time = std::chrono::high_resolution_clock::now();
    }

    ~Timer()
    {
        stop();
    }

    void stop()
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto start_us = std::chrono::time_point_cast<std::chrono::microseconds>(_start_time).time_since_epoch().count();
        auto end_us   = std::chrono::time_point_cast<std::chrono::microseconds>(end_time).time_since_epoch().count();
        _time_dst = static_cast<float>(end_us - start_us) / 1'000;
        if (_print)
            std::cout << _time_dst << '\n';
    }
};

#define TIMER(A) Timer timer##__LINE__(A, false)
#define TIMER_PRINT(A) Timer timer##__LINE__(A, true)
