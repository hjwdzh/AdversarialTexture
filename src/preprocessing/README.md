# Data Preprocessing

We prestore the depth, color and uv-image mapping for differentiable rendering and misaligned data augmentation.

### Compile
To preprocess the scan, first compile the cuda rendering and cpp libraries by
```
cd CudaRender
sh compile.sh
cd ../Rasterizer
sh compile.sh
```

### Render every scan
If you have multiple GPUs or large GPU memory, consider run the following in parallel.
```
sh run.sh
```
