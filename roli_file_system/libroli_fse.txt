#pragma once

#include <cstdint>

extern "C" {
    void* roli_fse_open(char* filename);
    int roli_fse_get_width(void* handle, uint32_t* width);
    int roli_fse_get_height(void* handle, uint32_t* height);
    int roli_fse_get_n_frames(void* handle, uint64_t* n_frames);
    int roli_fse_read(void* handle, uint16_t* img, int i);
    void roli_fse_close(void* handle);
}
