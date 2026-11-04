// ============================================================================
//  font5x7.h  --  5x7 bitmap font, just enough to stamp text into an image
// ----------------------------------------------------------------------------
//  Why bother: the rendered image is the only artefact of this project you can
//  actually look at, so stamping version / resolution / render time into the
//  corner turns every PNG into its own evidence ("this picture really came
//  from this build at this resolution").  It doubles as the debug overlay for
//  later weeks (hit counts, recursion depth, ...).
//
//  Encoding: every glyph is 5 bytes, one byte per COLUMN (bit 0 = top row,
//  bit 6 = bottom row) -- the classic "font5x7" column layout.
//  Marked __host__ __device__ so the same table can be used from a kernel or
//  from the host-side post-processing code.
// ============================================================================
#pragma once

// Returns the 5-byte column bitmap for `c`, or NULL if the glyph is unknown.
__host__ __device__ inline const unsigned char* glyph5x7(char c)
{
    switch (c)
    {
    // ---- digits ----
    case '0': { static const unsigned char g[5] = { 0x3E,0x51,0x49,0x45,0x3E }; return g; }
    case '1': { static const unsigned char g[5] = { 0x00,0x42,0x7F,0x40,0x00 }; return g; }
    case '2': { static const unsigned char g[5] = { 0x42,0x61,0x51,0x49,0x46 }; return g; }
    case '3': { static const unsigned char g[5] = { 0x21,0x41,0x45,0x4B,0x31 }; return g; }
    case '4': { static const unsigned char g[5] = { 0x18,0x14,0x12,0x7F,0x10 }; return g; }
    case '5': { static const unsigned char g[5] = { 0x27,0x45,0x45,0x45,0x39 }; return g; }
    case '6': { static const unsigned char g[5] = { 0x3C,0x4A,0x49,0x49,0x30 }; return g; }
    case '7': { static const unsigned char g[5] = { 0x01,0x71,0x09,0x05,0x03 }; return g; }
    case '8': { static const unsigned char g[5] = { 0x36,0x49,0x49,0x49,0x36 }; return g; }
    case '9': { static const unsigned char g[5] = { 0x06,0x49,0x49,0x29,0x1E }; return g; }

    // ---- uppercase ----
    case 'A': { static const unsigned char g[5] = { 0x7E,0x11,0x11,0x11,0x7E }; return g; }
    case 'B': { static const unsigned char g[5] = { 0x7F,0x49,0x49,0x49,0x36 }; return g; }
    case 'C': { static const unsigned char g[5] = { 0x3E,0x41,0x41,0x41,0x22 }; return g; }
    case 'D': { static const unsigned char g[5] = { 0x7F,0x41,0x41,0x22,0x1C }; return g; }
    case 'E': { static const unsigned char g[5] = { 0x7F,0x49,0x49,0x49,0x41 }; return g; }
    case 'F': { static const unsigned char g[5] = { 0x7F,0x09,0x09,0x09,0x01 }; return g; }
    case 'G': { static const unsigned char g[5] = { 0x3E,0x41,0x49,0x49,0x7A }; return g; }
    case 'H': { static const unsigned char g[5] = { 0x7F,0x08,0x08,0x08,0x7F }; return g; }
    case 'I': { static const unsigned char g[5] = { 0x00,0x41,0x7F,0x41,0x00 }; return g; }
    case 'J': { static const unsigned char g[5] = { 0x20,0x40,0x41,0x3F,0x01 }; return g; }
    case 'K': { static const unsigned char g[5] = { 0x7F,0x08,0x14,0x22,0x41 }; return g; }
    case 'L': { static const unsigned char g[5] = { 0x7F,0x40,0x40,0x40,0x40 }; return g; }
    case 'M': { static const unsigned char g[5] = { 0x7F,0x02,0x0C,0x02,0x7F }; return g; }
    case 'N': { static const unsigned char g[5] = { 0x7F,0x04,0x08,0x10,0x7F }; return g; }
    case 'O': { static const unsigned char g[5] = { 0x3E,0x41,0x41,0x41,0x3E }; return g; }
    case 'P': { static const unsigned char g[5] = { 0x7F,0x09,0x09,0x09,0x06 }; return g; }
    case 'Q': { static const unsigned char g[5] = { 0x3E,0x41,0x51,0x21,0x5E }; return g; }
    case 'R': { static const unsigned char g[5] = { 0x7F,0x09,0x19,0x29,0x46 }; return g; }
    case 'S': { static const unsigned char g[5] = { 0x46,0x49,0x49,0x49,0x31 }; return g; }
    case 'T': { static const unsigned char g[5] = { 0x01,0x01,0x7F,0x01,0x01 }; return g; }
    case 'U': { static const unsigned char g[5] = { 0x3F,0x40,0x40,0x40,0x3F }; return g; }
    case 'V': { static const unsigned char g[5] = { 0x1F,0x20,0x40,0x20,0x1F }; return g; }
    case 'W': { static const unsigned char g[5] = { 0x3F,0x40,0x38,0x40,0x3F }; return g; }
    case 'X': { static const unsigned char g[5] = { 0x63,0x14,0x08,0x14,0x63 }; return g; }
    case 'Y': { static const unsigned char g[5] = { 0x07,0x08,0x70,0x08,0x07 }; return g; }
    case 'Z': { static const unsigned char g[5] = { 0x61,0x51,0x49,0x45,0x43 }; return g; }

    // ---- symbols that show up in the stamps and debug overlays ----
    case ' ': { static const unsigned char g[5] = { 0x00,0x00,0x00,0x00,0x00 }; return g; }
    case '.': { static const unsigned char g[5] = { 0x00,0x60,0x60,0x00,0x00 }; return g; }
    case ',': { static const unsigned char g[5] = { 0x00,0x50,0x30,0x00,0x00 }; return g; }
    case ':': { static const unsigned char g[5] = { 0x00,0x36,0x36,0x00,0x00 }; return g; }
    case '-': { static const unsigned char g[5] = { 0x08,0x08,0x08,0x08,0x08 }; return g; }
    case '+': { static const unsigned char g[5] = { 0x08,0x3E,0x08,0x00,0x00 }; return g; }
    case '/': { static const unsigned char g[5] = { 0x20,0x10,0x08,0x04,0x02 }; return g; }
    case '=': { static const unsigned char g[5] = { 0x14,0x14,0x14,0x14,0x14 }; return g; }
    case '(': { static const unsigned char g[5] = { 0x00,0x1C,0x22,0x41,0x00 }; return g; }
    case ')': { static const unsigned char g[5] = { 0x00,0x41,0x22,0x1C,0x00 }; return g; }
    case '#': { static const unsigned char g[5] = { 0x14,0x7F,0x14,0x7F,0x14 }; return g; }
    default:  return NULL;
    }
}

// Pixels a glyph occupies horizontally (5 columns + 1 blank column).
#define FONT5X7_ADVANCE 6
// Glyph height in pixels (7 rows + 1 blank row).
#define FONT5X7_HEIGHT  8

// ---------------------------------------------------------------------------
//  stampTextHost  --  draw text into an 8-bit RGBA framebuffer, on the host
// ---------------------------------------------------------------------------
// Doing the stamp on the host matters: stamping inside the kernel means a
// second launch plus a second copy, which would distort the measured kernel
// time that we print on the image.
inline void stampTextHost(unsigned char* rgba, int width, int height,
                          int originX, int originY, const char* text,
                          unsigned char r, unsigned char g, unsigned char b)
{
    int cx = originX;
    for (const char* p = text; *p != '\0'; ++p)
    {
        const unsigned char* glyph = glyph5x7(*p);
        if (glyph != NULL)
        {
            for (int col = 0; col < 5; ++col)
            {
                const unsigned char bits = glyph[col];
                for (int row = 0; row < 7; ++row)
                {
                    if ((bits >> row) & 1u)
                    {
                        const int px = cx + col;
                        const int py = originY + row;
                        if (px >= 0 && px < width && py >= 0 && py < height)
                        {
                            unsigned char* dst = rgba + ((size_t)py * (size_t)width + (size_t)px) * 4u;
                            dst[0] = r; dst[1] = g; dst[2] = b; dst[3] = 255;
                        }
                    }
                }
            }
        }
        cx += FONT5X7_ADVANCE;
    }
}
