#ifndef RENDER_H_
#define RENDER_H_

#include "buffer.hpp"

void NaiveRender(FrameBuffer& frameBuffer);
void Render(VertexBuffer& vertexBuffer, FrameBuffer& frameBuffer, int renderPrimitive = 1);
void FetchDepth(FrameBuffer& frameBuffer);
void FetchVMap(VertexBuffer& vertexBuffer, FrameBuffer& frameBuffer);

#endif