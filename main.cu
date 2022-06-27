#include "src/external.h"
#include "src/utils/debug.h"
#include "src/utils/utils.h"
#include "src/utils/mtl_utils.h"
#include "src/utils/scene_builder.h"
#include "src/core/ray.h"
#include "src/core/camera.h"
#include "src/math/vector3D.h"
#include "src/geometries/sphere.h"
#include "src/geometries/object_list.h"
#include "src/material.h"
#include "src/kernels/render.h"
#include "src/loadOBJ.h"

void save_to_ppm(vector3D *fb, int WIDTH, int HEIGHT);
void save_to_jpg(vector3D *fb, int WIDTH, int HEIGHT, int i);
std::string toString(int &i);

__global__ void free_objects(object **d_list, int num_objects) {
    // for (int i = 0; i < num_objects; i++) {
    //     delete *(d_list + i);
    // }
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_objects) return;
    delete *(d_list + idx);
}

__global__ void free_world(object **d_world, camera **d_camera) {
    delete *d_world;
    delete *d_camera;
}

__global__ void free_others(material **d_mats, material **d_converted_mats, Texture** background) {
    delete *d_mats;
    delete *d_converted_mats;
    delete *background;
}

void get_device_props() {
    int nDevices;
  
    cudaGetDeviceCount(&nDevices);
    std::cerr << "+-----------------------------------------------------------------------------+" << std::endl;
    std::cerr << "                            CUDA DEVICE PROPERTIES" << std::endl;
    std::cerr << "+-----------------------------------------------------------------------------+" << std::endl;
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cerr << " Device Number: " << i << std::endl;
        std::cerr << "+=============================================================================+" << std::endl;
        std::cerr << " Device name: " << prop.name << std::endl;
        std::cerr << " Memory Clock Rate (KHz): "
                    << prop.memoryClockRate << std::endl;
        std::cerr << " Memory Bus Width (bits): "
                    << prop.memoryBusWidth << std::endl;
        std::cerr << " \tPeak Memory Bandwidth (GB/s): "
                    << 2.0f * prop.memoryClockRate *
                        (prop.memoryBusWidth / 8) / 1.0e6
                    << std::endl;
    }
    std::cerr << "+-----------------------------------------------------------------------------+" << std::endl;
}

__global__ void build_background(ImageData backgroundData, Texture** background) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (backgroundData.d_imgData != nullptr) {
        background[0] = new image_texture(backgroundData.d_imgData, backgroundData.width, backgroundData.height);
    } else {
        background[0] = new constant_texture(color(0.95f));
    }
}

__global__ void build_background(color colore, Texture** background) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    background[0] = new constant_texture(colore);
}

int main() {
    float aspect_ratio = 16.f / 9.f;
    int WIDTH = 1920;
    int HEIGHT = static_cast<int>(WIDTH / aspect_ratio);
    int SAMPLES = 100;
    int BLOCK_WIDTH = 8;
    int BLOCK_HEIGHT = 8;
    int BOUNCES = 5;

    get_device_props();

    std::cerr << "Rendering a " << WIDTH << "x" << HEIGHT << " image with " << SAMPLES << " samples per pixel ";
    std::cerr << "in " << BLOCK_WIDTH << "x" << BLOCK_HEIGHT << " blocks.\n";

    int num_pixels = WIDTH * HEIGHT;
    size_t fb_size = num_pixels*sizeof(vector3D);

    // -- allocate FB
    vector3D *fb;
    CUDA_CONTROL(cudaMallocManaged((void **)&fb, fb_size));
    // -- allocate random state
    curandState *d_rand_state;
    CUDA_CONTROL(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    CUDA_CONTROL(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // -- we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    // -- make our world of objects & the camera
    object **d_world;
    CUDA_CONTROL(cudaMalloc((void **)&d_world, sizeof(object *)));
    camera **d_camera;
    CUDA_CONTROL(cudaMalloc((void **)&d_camera, sizeof(camera *)));

    clock_t start, stop, start_total, stop_total;

    start_total = clock();
    start = clock();
    // -- render our buffer
    dim3 blocks(WIDTH / BLOCK_WIDTH + 1, HEIGHT / BLOCK_HEIGHT + 1);
    dim3 threads(BLOCK_WIDTH, BLOCK_HEIGHT);

    render_init<<<blocks, threads>>>(WIDTH, HEIGHT, d_rand_state);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    material **d_mats;
    CUDA_CONTROL(cudaMalloc((void **)&d_mats, sizeof(material *)));

    // mtl materials
    material **d_converted_mats;
    CUDA_CONTROL(cudaMalloc((void **)&d_converted_mats, sizeof(material *)));

    // ----------------------------- SKYBOX -----------------------------
#ifdef SCENE_SKYBOX
    SAMPLES = 100;

    object **d_list;
    int num_objects = 1;
    CUDA_CONTROL(cudaMalloc((void **)&d_list, num_objects * sizeof(object *)));

    // -- creazione scena
    build_scene_skybox(d_list, d_rand_state2, d_mats, d_world, num_objects);

    gpu_memory();

    // -- background (hdr)
    ImageData backgroundData = caricaImmagine("/content/ProgettoGraficaCUDA/skybox/city_night.jpg");
    Texture** background;
    CUDA_CONTROL(cudaMalloc((void **)&background, sizeof(Texture)));
    build_background<<<1, 1>>>(backgroundData, background);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    // -- creazione camera
    vector3D lookfrom(300,0,10);
    vector3D lookat(0,0,0);
    build_camera<<<1,1>>>(d_camera, WIDTH, HEIGHT, lookat, lookfrom, false);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    // -- render
    render<<<blocks, threads>>>(fb, WIDTH, HEIGHT, SAMPLES, BOUNCES, d_camera, d_world, d_rand_state, background);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "image rendered in " << timer_seconds << " seconds.\n";

    std::cerr << "image saving in progress\n";
    start = clock();
    
    save_to_jpg(fb, WIDTH, HEIGHT, 0);

    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    gpu_memory();
#endif
    // ---------------------------- !SKYBOX -----------------------------

    // ----------------------------- SPHERES -----------------------------
#ifdef SCENE_SPHERES
    SAMPLES = 100;
    BOUNCES = 5;

    object **d_list;
    int num_objects = 485;
    CUDA_CONTROL(cudaMalloc((void **)&d_list, num_objects * sizeof(object *)));

    // -- creazione scena
    build_scene_spheres(d_list, d_rand_state2, d_mats, d_world, num_objects);

    BVHNode* bvh_root = create_BVH(d_list, num_objects);
    
    gpu_memory();

    // -- background (hdr)
    ImageData backgroundData = caricaImmagine("/content/ProgettoGraficaCUDA/skybox/desert.jpeg");
    Texture** background;
    CUDA_CONTROL(cudaMalloc((void **)&background, sizeof(Texture)));
    build_background<<<1, 1>>>(backgroundData, background);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    // -- creazione camera
    vector3D lookfrom(13,2,3);
    vector3D lookat(0,0,0);
    build_camera<<<1,1>>>(d_camera, WIDTH, HEIGHT, lookat, lookfrom);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    // -- bvh render
    render<<<blocks, threads>>>(fb, WIDTH, HEIGHT, SAMPLES, BOUNCES, d_camera, bvh_root, d_rand_state, background);
    // -- normal render
    //render<<<blocks, threads>>>(fb, WIDTH, HEIGHT, SAMPLES, BOUNCES, d_camera, d_world, d_rand_state, background);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "image rendered in " << timer_seconds << " seconds.\n";

    std::cerr << "image saving in progress\n";
    start = clock();
    
    save_to_jpg(fb, WIDTH, HEIGHT, 0);

    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    gpu_memory();
#endif
    // ---------------------------- !SPHERES -----------------------------

    // ----------------------------- CORNEL BOX -----------------------------
#ifdef SCENE_CORNELL_BOX
    SAMPLES = 300;
    object **d_list;
    int num_objects = 8;
    CUDA_CONTROL(cudaMalloc((void **)&d_list, num_objects * sizeof(object *)));

    // -- creazione scena
    build_scene_cornel(d_list, d_rand_state2, d_world, num_objects);

    //BVHNode* bvh_root = create_BVH(d_list, num_objects);
    
    gpu_memory();

    // -- background (hdr)
    ImageData backgroundData = caricaImmagine("/content/ProgettoGraficaCUDA/skybox/desert.jpeg");
    Texture** background;
    CUDA_CONTROL(cudaMalloc((void **)&background, sizeof(Texture)));
    build_background<<<1, 1>>>(backgroundData, background);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    // -- creazione camera
    point3D lookfrom(278, 278, -800);
    point3D lookat(278, 278, 0);
    build_camera<<<1,1>>>(d_camera, WIDTH, HEIGHT, lookat, lookfrom, false);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    // -- bvh render
    //render<<<blocks, threads>>>(fb, WIDTH, HEIGHT, SAMPLES, BOUNCES, d_camera, bvh_root, d_rand_state, background);
    // -- normal render
    render<<<blocks, threads>>>(fb, WIDTH, HEIGHT, SAMPLES, BOUNCES, d_camera, d_world, d_rand_state, background);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "image rendered in " << timer_seconds << " seconds.\n";

    std::cerr << "image saving in progress\n";
    start = clock();
    
    save_to_jpg(fb, WIDTH, HEIGHT, 0);

    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    gpu_memory();
#endif
    // ---------------------------- !CORNEL BOX -----------------------------

    // ----------------------------- STAR_WARS -----------------------------
#ifdef SCENE_STAR_WARS
    objData obj_0 = load_obj("/content/ProgettoGraficaCUDA/models/R2D2/Low_Poly_R2D2.obj", "/content/ProgettoGraficaCUDA/models/R2D2/");
    objData obj_1 = load_obj("/content/ProgettoGraficaCUDA/models/Light_Saber/circle.obj");
    objData obj_2 = load_obj("/content/ProgettoGraficaCUDA/models/Light_Saber/light.obj");
    objData obj_3 = load_obj("/content/ProgettoGraficaCUDA/models/Light_Saber/handle_terrain.obj");
    objData obj_4 = load_obj("/content/ProgettoGraficaCUDA/models/Light_Saber/light_terrain.obj");
    objData obj_5 = load_obj("/content/ProgettoGraficaCUDA/models/Helmet_Stormtrooper/Helmet_Stormtrooper.obj");
    objData obj_6 = load_obj("/content/ProgettoGraficaCUDA/models/c3po.obj");
    objData obj_7 = load_obj("/content/ProgettoGraficaCUDA/models/Farming_Droid/Farming_Droid_2.obj");
    
    set_1GB_heap_size();

    object **d_list;
    int num_objects = (obj_0.num_triangles + obj_1.num_triangles + obj_2.num_triangles + obj_3.num_triangles + obj_4.num_triangles + obj_5.num_triangles + obj_6.num_triangles + obj_7.num_triangles) + 1 + 1; // objs + floor + sphere
    printf("[main] • num_objects: %d\n", num_objects);
    CUDA_CONTROL(cudaMalloc((void **)&d_list, num_objects * sizeof(object *)));

    objData objs[8] = { obj_0, obj_1, obj_2, obj_3, obj_4, obj_5, obj_6, obj_7 };

    // -- creazione scena
    build_scene_star_wars(d_list, objs, d_rand_state2, d_mats, d_converted_mats);
    BVHNode* bvh_root = create_BVH(d_list, num_objects);

    gpu_memory();

    start = clock();

    // -- background (hdr)
    ImageData backgroundData = caricaImmagine("/content/ProgettoGraficaCUDA/skybox/desert.jpeg");
    Texture** background;
    CUDA_CONTROL(cudaMalloc((void **)&background, sizeof(Texture)));
    build_background<<<1, 1>>>(backgroundData, background);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    // -- creazione camera
    point3D lookat = vector3D(30,0,2);
    point3D lookfrom = vector3D(0,5,0);
    //point3D lookfrom(13,2,3);
    //point3D lookat(0,0,0);
    build_camera<<<1,1>>>(d_camera, WIDTH, HEIGHT, lookat, lookfrom);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    // -- bvh render
    render<<<blocks, threads>>>(fb, WIDTH, HEIGHT, SAMPLES, BOUNCES, d_camera, bvh_root, d_rand_state, background);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "image rendered in " << timer_seconds << " seconds.\n";

    std::cerr << "image saving in progress\n";
    start = clock();
    // -- output FB as Image
    // save_to_ppm(fb, WIDTH, HEIGHT);
    save_to_jpg(fb, WIDTH, HEIGHT, 0);

    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    gpu_memory();
#endif
    // ---------------------------- !STAR_WARS -----------------------------

    // ----------------------------- GALLERY -----------------------------
#ifdef SCENE_GALLERY
    objData obj_0 = load_obj("/content/ProgettoGraficaCUDA/models/Helmet_Stormtrooper/Helmet_Stormtrooper.obj");
    objData obj_1 = load_obj("/content/ProgettoGraficaCUDA/models/bender.obj");
    //objData obj_1 = load_obj("/content/ProgettoGraficaCUDA/models/robot_cat_sketchfab.obj");
    objData obj_2 = load_obj("/content/ProgettoGraficaCUDA/models/skull.obj");
    objData obj_3 = load_obj("/content/ProgettoGraficaCUDA/models/origami_cat.obj");
    objData obj_4 = load_obj("/content/ProgettoGraficaCUDA/models/origami_dog.obj");
    objData obj_5 = load_obj("/content/ProgettoGraficaCUDA/models/raptor.obj");
    objData obj_6 = load_obj("/content/ProgettoGraficaCUDA/models/whale.obj");
    objData obj_7 = load_obj("/content/ProgettoGraficaCUDA/models/Mandalorian.obj");
    
    // get_heap_size();
    set_1GB_heap_size();
    // get_heap_size();

    object **d_list;
    int num_objects = (obj_0.num_triangles + obj_1.num_triangles + obj_2.num_triangles + obj_3.num_triangles + obj_4.num_triangles + obj_5.num_triangles + obj_6.num_triangles + obj_7.num_triangles) + 1 + 1; // objs + plane + cyl
    printf("[main] • num_objects: %d\n", num_objects);
    CUDA_CONTROL(cudaMalloc((void **)&d_list, num_objects * sizeof(object *)));

    objData objs[8] = { obj_0, obj_1, obj_2, obj_3, obj_4, obj_5, obj_6, obj_7 };

    // -- creazione scena
    build_scene_gallery(d_list, objs, d_rand_state2, d_mats, d_converted_mats);
    BVHNode* bvh_root = create_BVH(d_list, num_objects);

    gpu_memory();

    vector3D lookfrom; // z,x,y
    vector3D lookat;
    
    // -- background (hdr)
    ImageData backgroundData = caricaImmagine("/content/ProgettoGraficaCUDA/skybox/desert2.jpeg");
    Texture** background;
    CUDA_CONTROL(cudaMalloc((void **)&background, sizeof(Texture)));
    build_background<<<1, 1>>>(backgroundData, background);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    // ----------> ROTAZIONE CAMERA (render multipli)
    // - camera a terra
    //lookfrom = vector3D(15,0,3);
    //lookat = vector3D(0,1,0);
    lookfrom = vector3D(15,0,2);
    lookat = vector3D(0,5,0);
    
    int num = 72;
    for (int i = 0; i < num; i++) {
        start = clock();
        float degree = i * (360 / num);
        point3D newPoint = rotateY(lookfrom, lookat, degree_to_radian(degree));
        //std::cerr << "newPoint: [" << newPoint.x() << ", " << newPoint.y() << ", " << newPoint.z() << "] - angle: " << degree << "°" << std::endl;

        build_camera<<<1,1>>>(d_camera, WIDTH, HEIGHT, newPoint, lookat);
        CUDA_CONTROL(cudaGetLastError());
        CUDA_CONTROL(cudaDeviceSynchronize());

        // -- bvh render
        render<<<blocks, threads>>>(fb, WIDTH, HEIGHT, SAMPLES, BOUNCES, d_camera, bvh_root, d_rand_state, background);
        CUDA_CONTROL(cudaGetLastError());
        CUDA_CONTROL(cudaDeviceSynchronize());

        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::cerr << "image rendered in " << timer_seconds << " seconds.\n";

        std::cerr << "image saving in progress\n";
        start = clock();
        // -- output FB as Image
        // save_to_ppm(fb, WIDTH, HEIGHT);
        save_to_jpg(fb, WIDTH, HEIGHT, i);

        stop = clock();
        timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::cerr << "took " << timer_seconds << " seconds.\n";

        gpu_memory();
    }
#endif
    // ---------------------------- !GALLERY -----------------------------

    stop_total = clock();
    double total_time = ((double)(stop_total - start_total)) / CLOCKS_PER_SEC;
    std::cerr << "in total it took " << total_time << " seconds.\n";

    // -- clean up
    CUDA_CONTROL(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_world, d_camera);
    CUDA_CONTROL(cudaGetLastError());
    free_objects<<<1,1>>>(d_list, num_objects);
    CUDA_CONTROL(cudaGetLastError());
    free_others<<<1,1>>>(d_mats, d_converted_mats, background);
    CUDA_CONTROL(cudaFree(d_camera));
    CUDA_CONTROL(cudaFree(d_world));
    CUDA_CONTROL(cudaFree(d_list));
    CUDA_CONTROL(cudaFree(d_rand_state));
    CUDA_CONTROL(cudaFree(d_rand_state2));
    CUDA_CONTROL(cudaFree(fb));
    CUDA_CONTROL(cudaFree(d_mats));
    CUDA_CONTROL(cudaFree(d_converted_mats));
    CUDA_CONTROL(cudaFree(background));
    CUDA_CONTROL(cudaGetLastError());

    cudaDeviceReset();
}

void save_to_ppm(vector3D *fb, int WIDTH, int HEIGHT) {
	std::ofstream ofs;
	ofs.open("image.ppm", std::ios::out | std::ios::binary);
	ofs << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (int j = HEIGHT - 1; j >= 0; j--) {
        for (int i = 0; i < WIDTH; i++) {
            size_t pixel_index = j * WIDTH + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            ofs << ir << " " << ig << " " << ib << "\n";
        }
    }
	ofs.close();
}

void save_to_jpg(vector3D *fb, int WIDTH, int HEIGHT, int i) {
    uint8_t* imgBuff = (uint8_t*)std::malloc(WIDTH * HEIGHT * 3 * sizeof(uint8_t));
    for (int j = HEIGHT - 1; j >= 0; j--) {
        for (int i = 0; i < WIDTH; i++) {
            size_t index = j * WIDTH + i;
            float r = fb[index].r();
            float g = fb[index].g();
            float b = fb[index].b();
            // stbi generates a Y flipped image
            size_t rev_index = (HEIGHT - j - 1) * WIDTH + i;
            imgBuff[rev_index * 3 + 0] = int(255.999f * r) & 255;
            imgBuff[rev_index * 3 + 1] = int(255.999f * g) & 255;
            imgBuff[rev_index * 3 + 2] = int(255.999f * b) & 255;
        }
    }
    //stbi_write_png("out.png", WIDTH, HEIGHT, 3, imgBuff, WIDTH * 3);
    std::string str = "image_" + (i < 10 ? "0" + std::to_string(i) : std::to_string(i)) + ".jpg";
    stbi_write_jpg(str.c_str(), WIDTH, HEIGHT, 3, imgBuff, 100);
    std::free(imgBuff);
}

std::string toString(int &i) {
    std::stringstream ss;
    ss << i;
    return ss.str();
}
