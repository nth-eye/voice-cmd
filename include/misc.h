#include "model_settings.h"

#pragma once

// std::array implementation to allow return by value and range-based for loop.
template<class T, size_t N>
class Array {
    T data[N] = {};
public:
    constexpr size_t size() { return N; }

    T& operator[](size_t idx)   { return data[idx]; }
    T* begin()                  { return &data[0]; }
    T* end()                    { return &data[N]; }
    const T& operator[](size_t idx) const   { return data[idx]; }
    const T* begin() const                  { return &data[0]; }
    const T* end() const                    { return &data[N]; }
};

// Data structure that holds an inference result and the time when it was recorded.
struct Result {
    Result() = default;
    Result(int32_t time_, const Array<int8_t, N_LABELS> &scores_) : time(time_), scores(scores_) {}
    Result(int32_t time_, int8_t *scores_) : time(time_) 
    {
        for (size_t i = 0; i < N_LABELS; ++i)
            scores[i] = scores_[i];
    }

    int32_t time = 0;
    Array<int8_t, N_LABELS> scores = {};
};

// Struct which holds found command, its confidence score and is it same as previous.
struct Command {
    uint8_t found_command = SILENCE;
    uint8_t score = 0;
    bool is_new = false;
};

// Circular buffer implementation. Uses N-1 elements logic.
template<class T, size_t N>
class RingBuf {

    T buf[N] = {};
    size_t head = 0;    // First item index / beginning of the buffer.
    size_t tail = 0;    // Last item index.

    class iterator {
        RingBuf<T, N> &buf;
        size_t pos;
    public: 
        iterator(RingBuf<T, N> &buf_, size_t pos_) : buf(buf_), pos(pos_) {}

        T& operator*()  { return buf[pos]; }
        T* operator->() { return &(operator*()); }
        iterator& operator++() 
        { 
            if (++pos == N) 
                pos = 0;
            return *this;
        }
        iterator operator++(int) 
        {   
            T tmp(*this);
            operator++();
            return tmp;
        }
        bool operator==(const iterator &other) { return pos == other.pos; }
        bool operator!=(const iterator &other) { return !operator==(other); }
    };
public:
    iterator begin()            { return iterator(*this, head); }
    iterator end()              { return iterator(*this, tail); }
    T& operator[](size_t idx)   { return buf[idx]; }
    T& front()                  { return operator[](head); }
    T& back()                   { return operator[](tail); }
    const iterator begin() const            { return iterator(*this, head); }
    const iterator end() const              { return iterator(*this, tail); }
    const T& operator[](size_t idx) const   { return buf[idx]; }
    const T& front() const                  { return operator[](head); }
    const T& back() const                   { return operator[](tail); }
    // Responsibility to check empty() is on the caller.
    void pop_front()                    { if (++head == N) head = 0; }
    void push_back(const T &item)
    {
        size_t next = tail + 1;
        if (next == N) next = 0;
        if (next == head) return;
        buf[tail] = item;
        tail = next;
    }
    void clear()                        { head = tail = 0; }
    bool empty() const                  { return tail == head; }
    size_t size() const                 { return tail > head ? tail - head: N + tail - head; }
    size_t constexpr capacity() const   { return N - 1; }
};
