// ============================================================================
//  png_writer.h  --  thin wrapper around the provided stb_image_write.h
// ----------------------------------------------------------------------------
//  stb_image_write.h implements
//      stbi_write_png(filename, w, h, comp, data, stride)
//  and lives in `Whitted\可能会用到的库\`.  This wrapper only:
//    * picks the stb file up (next to kernel.cu, or from the repo folder),
//    * defines the implementation macro in exactly one place,
//    * hides the "header missing" case behind a bool so the program still
//      builds and keeps writing PPM.
//
//  Search order (first hit wins):
//    1. stb_image_write.h copied next to kernel.cu  -> self-contained project
//    2. Whitted\可能会用到的库\stb_image_write.h   -> resolved relative to this file
//    3. nothing -> PNG disabled, PPM still written
//
//  `rgb` must be w*h*3 bytes, row-major top-to-bottom, R,G,B order (which is
//  what the uchar4 render buffer reduced to RGB gives you).
// ============================================================================
#pragma once

#include <stdio.h>

#if defined(__has_include)
#  if __has_include("stb_image_write.h")
#    define RT_HAVE_STB 1
#    define RT_STB_PATH "stb_image_write.h"
#  elif __has_include("../_third_party/stb/stb_image_write.h")
#    define RT_HAVE_STB 1
#    define RT_STB_PATH "../_third_party/stb/stb_image_write.h"
#  else
#    define RT_HAVE_STB 0
#  endif
#else
#  define RT_HAVE_STB 0
#endif

#if RT_HAVE_STB
#  ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#    define STB_IMAGE_WRITE_IMPLEMENTATION
#  endif
#  include RT_STB_PATH
#endif

// Returns true when the program is able to write PNG files.
inline bool pngWriterAvailable()
{
#if RT_HAVE_STB
    return true;
#else
    return false;
#endif
}

// Writes an 8-bit RGB image (3 bytes per pixel, top row first).
// Returns true on success.
inline bool writePngRGB(const char* path, int width, int height, const unsigned char* rgb, int compressionLevel = 6)
{
#if RT_HAVE_STB
    stbi_write_png_compression_level = compressionLevel;
    return stbi_write_png(path, width, height, 3, rgb, width * 3) != 0;
#else
    (void)path; (void)width; (void)height; (void)rgb; (void)compressionLevel;
    fprintf(stderr, "[png_writer] stb_image_write.h not found -- PNG output disabled, PPM still written.\n");
    return false;
#endif
}
