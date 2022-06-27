#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <thrust/copy.h>
#include <thrust/device_delete.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/swap.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

// stb image read & write
#define STB_IMAGE_IMPLEMENTATION
#include "../libs/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../libs/stb/stb_image_write.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "../libs/tiny_obj_loader.h"
