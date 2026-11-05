// ============================================================================
//  image_io.h  --  PPM (P6) + PNG output, plus the pixel-diff used to prove
//                  that an optimisation did not change the image
// ----------------------------------------------------------------------------
//  Debug loop recommendation (from task-all.md): write PPM and open it in the
//  Windows photo viewer.  It is a handful of lines of code, has zero
//  dependencies and is fast even for 4K images, so it stays the primary
//  format.  PNG is written next to it because it is what you put in a README.
// ============================================================================
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "png_writer.h"

// ---------------------------------------------------------------------------
//  filesystem helpers (Win32, no <filesystem> so the code stays C++11 clean)
// ---------------------------------------------------------------------------
#ifdef _WIN32
#  include <direct.h>
#  include <sys/stat.h>
#  define RT_MKDIR(p) _mkdir(p)
#else
#  include <sys/stat.h>
#  include <sys/types.h>
#  define RT_MKDIR(p) mkdir((p), 0755)
#endif

// Creates `dir` if it does not exist yet. Returns true if it exists afterwards.
inline bool ensureDirectory(const char* dir)
{
    if (dir == NULL || dir[0] == '\0') return false;
#ifdef _WIN32
    struct _stat st;
    if (_stat(dir, &st) == 0) return (st.st_mode & _S_IFDIR) != 0;
#else
    struct stat st;
    if (stat(dir, &st) == 0) return S_ISDIR(st.st_mode);
#endif
    RT_MKDIR(dir);
#ifdef _WIN32
    if (_stat(dir, &st) == 0) return (st.st_mode & _S_IFDIR) != 0;
#else
    if (stat(dir, &st) == 0) return S_ISDIR(st.st_mode);
#endif
    return false;
}

// ---------------------------------------------------------------------------
//  filename convention:  weekN_<w>x<h>spp<s>[_tag].<ext>
// ---------------------------------------------------------------------------
inline void buildOutputStem(char* out, int outSize, const char* weekTag,
                            int width, int height, int spp, const char* tag)
{
    if (tag != NULL && tag[0] != '\0')
        snprintf(out, outSize, "%s_%dx%dspp%d_%s", weekTag, width, height, spp, tag);
    else
        snprintf(out, outSize, "%s_%dx%dspp%d", weekTag, width, height, spp);
}

// ---------------------------------------------------------------------------
//  PPM (P6 binary): 15 bytes of header + 3 bytes per pixel
// ---------------------------------------------------------------------------
// `rgba` is w*h*4 bytes (uchar4 layout, R,G,B,A); the alpha channel is dropped.
inline bool writePpmP6(const char* path, int width, int height, const unsigned char* rgba)
{
    FILE* f = NULL;
    if (fopen_s(&f, path, "wb") != 0 || f == NULL)
    {
        fprintf(stderr, "[image_io] cannot open for writing: %s\n", path);
        return false;
    }

    fprintf(f, "P6\n%d %d\n255\n", width, height);

    const size_t rowBytes = (size_t)width * 3u;
    unsigned char* row = (unsigned char*)malloc(rowBytes);
    if (row == NULL) { fclose(f); return false; }

    for (int y = 0; y < height; ++y)
    {
        const unsigned char* src = rgba + (size_t)y * (size_t)width * 4u;
        for (int x = 0; x < width; ++x)
        {
            row[x * 3 + 0] = src[x * 4 + 0];
            row[x * 3 + 1] = src[x * 4 + 1];
            row[x * 3 + 2] = src[x * 4 + 2];
        }
        fwrite(row, 1, rowBytes, f);
    }

    free(row);
    fclose(f);
    return true;
}

// Converts an interleaved RGBA buffer to tightly packed RGB.
inline void rgbaToRgb(const unsigned char* rgba, unsigned char* rgb, int width, int height)
{
    const size_t pixels = (size_t)width * (size_t)height;
    for (size_t i = 0; i < pixels; ++i)
    {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
}

inline bool writePngRGBA(const char* path, int width, int height, const unsigned char* rgba, int compressionLevel = 6)
{
    const size_t pixels = (size_t)width * (size_t)height;
    unsigned char* rgb = (unsigned char*)malloc(pixels * 3u);
    if (rgb == NULL) return false;
    rgbaToRgb(rgba, rgb, width, height);
    const bool ok = writePngRGB(path, width, height, rgb, compressionLevel);
    free(rgb);
    return ok;
}

// ---------------------------------------------------------------------------
//  the two calls the weekly kernels actually use
// ---------------------------------------------------------------------------
// Writes `<outDir>/<stem>.ppm` and `<outDir>/<stem>.png`.
// Returns the number of files successfully written (0, 1 or 2).
inline int writeImageBoth(const char* outDir, const char* stem,
                          int width, int height, const unsigned char* rgba,
                          bool wantPpm = true, bool wantPng = true)
{
    if (outDir != NULL) ensureDirectory(outDir);

    char path[1024];
    int written = 0;

    if (wantPpm)
    {
        snprintf(path, sizeof(path), "%s/%s.ppm", outDir ? outDir : ".", stem);
        if (writePpmP6(path, width, height, rgba))
        {
            ++written;
            printf("  saved  %s\n", path);
        }
    }
    if (wantPng)
    {
        snprintf(path, sizeof(path), "%s/%s.png", outDir ? outDir : ".", stem);
        if (pngWriterAvailable() && writePngRGBA(path, width, height, rgba))
        {
            ++written;
            printf("  saved  %s\n", path);
        }
    }
    return written;
}

// ---------------------------------------------------------------------------
//  image comparison -- the acceptance test for the BVH tiers
// ---------------------------------------------------------------------------
// Compares two RGBA buffers of the same size and reports the worst channel
// difference and the mean absolute difference.  "maxDiff == 0" is exactly the
// statement "the BVH did not change a single pixel", which is the strongest
// correctness argument you can put in front of an interviewer.
struct ImageDiff
{
    int   maxDiff;
    double meanDiff;
    long  pixelsDiffering;
};

inline ImageDiff compareImages(const unsigned char* a, const unsigned char* b,
                               int width, int height, bool ignoreAlpha = true)
{
    ImageDiff d;
    d.maxDiff = 0;
    d.meanDiff = 0.0;
    d.pixelsDiffering = 0;

    const int channels = ignoreAlpha ? 3 : 4;
    const size_t pixels = (size_t)width * (size_t)height;
    double sum = 0.0;

    for (size_t i = 0; i < pixels; ++i)
    {
        int pixelMax = 0;
        for (int c = 0; c < channels; ++c)
        {
            const int diff = (int)a[i * 4 + c] - (int)b[i * 4 + c];
            const int absDiff = diff < 0 ? -diff : diff;
            if (absDiff > pixelMax) pixelMax = absDiff;
            sum += (double)absDiff;
        }
        if (pixelMax > d.maxDiff) d.maxDiff = pixelMax;
        if (pixelMax != 0) ++d.pixelsDiffering;
    }

    d.meanDiff = sum / (double)(pixels * (size_t)channels);
    return d;
}

// Simple 2x2 box downsample (useful later to sanity check resolution changes).
inline void downsample2x2(const unsigned char* src, int width, int height, unsigned char* dst)
{
    const int w2 = width / 2;
    const int h2 = height / 2;
    for (int y = 0; y < h2; ++y)
    {
        for (int x = 0; x < w2; ++x)
        {
            int acc[4] = { 0, 0, 0, 0 };
            for (int dy = 0; dy < 2; ++dy)
            {
                for (int dx = 0; dx < 2; ++dx)
                {
                    const unsigned char* p = src + (((size_t)(y * 2 + dy) * width) + (x * 2 + dx)) * 4;
                    for (int c = 0; c < 4; ++c) acc[c] += p[c];
                }
            }
            unsigned char* q = dst + ((size_t)y * w2 + x) * 4;
            for (int c = 0; c < 4; ++c) q[c] = (unsigned char)(acc[c] / 4);
        }
    }
}
