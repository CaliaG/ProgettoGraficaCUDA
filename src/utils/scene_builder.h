#pragma once

#include "../loadOBJ.h"
#include "../core/camera.h"
#include "../geometries/object_list.h"
#include "../geometries/sphere.h"
#include "../geometries/cylinder.h"
#include "../geometries/aarect.h"
#include "../geometries/box2.h"
#include "../geometries/box.h"
#include "../utils/mtl_utils.h"

//#define SCENE_SKYBOX
//#define SCENE_SPHERES
//#define SCENE_CORNELL_BOX
//#define SCENE_STAR_WARS
#define SCENE_GALLERY

// ------------------> METODI DI UTILITA'
enum MaterialType {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    EMITTER,
    CHECKER,
    TEXTURE,
    NOISE,
    WOOD,
    GLOSSY
};

__global__ void build_camera(camera **d_camera, int WIDTH, int HEIGHT, point3D lookat, point3D lookfrom, bool isDepth = true) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // -- inizializzazione camera
    vector3D up(0,1,0);
    float fov = 30.0f;
    //*d_camera = new camera(lookfrom, lookat, up, fov, float(WIDTH)/float(HEIGHT));

    if (isDepth) {
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera_depth(lookfrom, lookat, up, fov, float(WIDTH)/float(HEIGHT), aperture, dist_to_focus);
    } else {
        *d_camera   = new camera(lookfrom, lookat, up, fov, float(WIDTH)/float(HEIGHT));
    }
}

__global__ void init_world(object **d_list, object **d_world, int num_objects) {
    *d_world  = new object_list(d_list, num_objects);
}

__device__ void initMaterial(material **d_mat, int matIdx, int idx, ImageData baseTextureData, color colore1 = color(0.8f, .0f, .0f), color colore2 = color(0.9f)) {
    switch (matIdx) {
		case MaterialType::LAMBERTIAN: d_mat[idx] = new lambertian(colore1); break;
		case MaterialType::METAL: d_mat[idx] = new metal(colore1, 0.05); break;
		case MaterialType::DIELECTRIC: d_mat[idx] = new dielectric(1.5); break;
		case MaterialType::EMITTER: d_mat[idx] = new emitter(new constant_texture(colore1), 1.f); break;
		case MaterialType::CHECKER: d_mat[idx] = new lambertianTexture(new checker_texture(new constant_texture(colore1), new constant_texture(colore2))); break;
        case MaterialType::TEXTURE: d_mat[idx] = new lambertianTexture(new image_texture(baseTextureData.d_imgData, baseTextureData.width, baseTextureData.height)); break;
        // TODO: finire di implementare > per ora sempre MARBLE
        case MaterialType::NOISE: d_mat[idx] = new lambertianTexture(new noise_texture(noise_type::MARBLE)); break;
        // TODO: finire di implementare > per ora colori predefiniti
        case MaterialType::WOOD: d_mat[idx] = new lambertianTexture(new wood_texture(color(.5f, .5f, .5f), color(.2f, .2f, .2f))); break;
        case MaterialType::GLOSSY: d_mat[idx] = new glossy(new constant_texture(colore1), new constant_texture(colore2)); break;
		default: d_mat[idx] = new lambertian(color(0.5, 0.5, 0.5));
	}
}

__device__ void create_material(material **d_mat, curandState *rand_state, int matIdx, int idx, color colore, objData obj, material **d_converted_mats, ImageData baseTextureData, MTLTextureData* mtlTexturesData
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    curandState local_rand_state = *rand_state;

    if (obj.mtl_loaded) {
        for (int i = 0; i < obj.num_materials; i++) {
            printf("• aggiungo mtl N°%d\n", i);
            mtlData mtl = obj.mtl[i];

            if (mtl.diffuse_texname[0] != '\0') {
                printf("• found image texture for mtl N°%d\n", i);
                ImageData diffuseData = mtlTexturesData[i].diffuseTex;
                Texture* diffuse_a = new image_texture(diffuseData.d_imgData, diffuseData.width, diffuseData.height);

                ImageData specularData = mtlTexturesData[i].specularTex;
                Texture* specular_a = new image_texture(specularData.d_imgData, specularData.width, specularData.height);

                Texture* emissive_a = new constant_texture(color(mtl.emissive_a[0], mtl.emissive_a[1], mtl.emissive_a[2]));
                Texture* transparency_a = new constant_texture(color(mtl.transmittance[0], mtl.transmittance[1], mtl.transmittance[2]) * (1. - mtl.dissolve));
                Texture* sharpness_a = new constant_texture(color(1, 0, 0) * mtl.shininess);
                d_converted_mats[i] = new mtl_material(diffuse_a, specular_a, emissive_a, transparency_a, sharpness_a, mtl.illum, RND);
            } else {
                printf("• no image texture\n");
                Texture* diffuse_a = new constant_texture(color(mtl.diffuse_a[0], mtl.diffuse_a[1], mtl.diffuse_a[2]));
                Texture* specular_a = new constant_texture(color(mtl.specular_a[0], mtl.specular_a[1], mtl.specular_a[2]));
                Texture* emissive_a = new constant_texture(color(mtl.emissive_a[0], mtl.emissive_a[1], mtl.emissive_a[2]));
                Texture* transparency_a = new constant_texture(color(mtl.transmittance[0], mtl.transmittance[1], mtl.transmittance[2]) * (1. - mtl.dissolve));
                Texture* sharpness_a = new constant_texture(color(1, 0, 0) * mtl.shininess);
                d_converted_mats[i] = new mtl_material(diffuse_a, specular_a, emissive_a, transparency_a, sharpness_a, mtl.illum, RND);
            }

        }
    }

    initMaterial(d_mat, matIdx, idx, baseTextureData, colore);

    *rand_state = local_rand_state;
}

__global__ void build_mat_obj(material **d_mat, curandState *rand_state, int matIdx, int idx, color colore, objData obj, material **d_converted_mats, ImageData baseTextureData, MTLTextureData* mtlTexturesData
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    create_material(d_mat, rand_state, matIdx, idx, colore, obj, d_converted_mats, baseTextureData, mtlTexturesData);
}

void allocate_obj(object **d_list, objData obj, int matIdx, curandState *rand_state, int start_id, vector3D& translate, float rotate, float scale, material **d_mats, material **d_converted_mats) {
	int obj_threads = 512;
	int obj_dims = (obj.num_triangles + obj_threads - 1) / obj_threads;

	create_obj_hittables<<<obj_dims, obj_threads>>>(d_list, matIdx, obj, rand_state, start_id, scale, translate, rotate, d_mats, d_converted_mats);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());
}

struct Statua {
    objData obj;
    vector3D translate;
    float rotate;
    float scale;
    color colore;
    int matIdx;
    // texture
    ImageData baseTextureData;
	// mtl - texture
    MTLTextureData* mtlTexturesData;
};

Statua creaStatua(objData obj, vector3D translate, float rotate, float scale, color colore, int matIdx, char imagePath[] = nullptr) {
    MTLTextureData* mtlTexturesData;
    if (obj.mtl_loaded) {
        mtlTexturesData = caricaTextureMTL(obj);
    }
    ImageData baseTexData;
    if (imagePath != nullptr) {
        /** carica texture (come float) */
        baseTexData = caricaImmagine(imagePath);
    }
    return { obj, translate, rotate, scale, colore, matIdx, baseTexData, mtlTexturesData };
}
// <------------------ METODI DI UTILITA'

// ==================== GALLERY ====================
__global__ void create_plane(object** d_list, int start_id, curandState *rand_state) {
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // - colore unico
    //d_list[start_id] = new sphere(vector3D(0,-1000.0,-1), 1000, new lambertian(color(0.5, 0.5, 0.5)));
    // - scacchiera
    //d_list[start_id] = new sphere(vector3D(0,-1000.0,-1), 1000, new lambertianTexture(new checker_texture(new constant_texture(color(.5f, .5f, .5f)), new constant_texture(color(0.9f, 0.9f, 0.9f)))));
    // - marmo
    d_list[start_id] = new sphere(vector3D(0,-1000.0,-1), 1000, new lambertianTexture(new noise_texture(noise_type::TURBULANCE)));
    // - legno
    //float density = 10.f;
    //float hardness = 100.f;
    //d_list[start_id] = new sphere(vector3D(0,-1000.0,-1), 1000, new lambertianTexture(new wood_texture(color(1, 193.f/255.f, 140.f/255.f), color(218.f/255.f, 109.f/255.f, 66.f/255.f), density, hardness)));
}

void create_statues(object **d_list, objData *objs, curandState *rand_state, int start_id, material **d_mats, material **d_converted_mats) {
    Statua statues[] = {
        //{ objs[0], vector3D(14, 0, 0),   -90.f,  .5f, color(1.0, 1.0, 1.0), MaterialType::TEXTURE, imgData_d, w, h },                   // dx (garfield)
        creaStatua(objs[0], vector3D(14, 3, 0),   -90.f,  .2f, color(.95), MaterialType::TEXTURE, "models/Helmet_Stormtrooper/core_BaseColor.png"),                   // dx (helmet stormtrooper)
        //creaStatua(objs[0], vector3D(14, 0, 0),   -90.f,  .2f, color(1.0, 1.0, 1.0), MaterialType::METAL),                              // (robot_cat_sketchfab)
        //objs[1], vector3D(9, 0, 9),    225.f,  .3f, color(0.5, 0.5, 0.5), MaterialType::METAL },                                      // dx -45° (bender)
        creaStatua(objs[1], vector3D(9, 0, 9),   225.f,  .3f, color(0.5f), MaterialType::METAL),                                        // dx -45° (bender)
        //creaStatua(objs[1], vector3D(9, 0, 9),   -90.f,  .2f, color(1.0, 1.0, 1.0), MaterialType::METAL),                               // (robot_cat_sketchfab)
        { objs[2], vector3D(9, 0, -9),    45.f, .18f, color(0.5, 0.5, 0.5), MaterialType::NOISE },                                      // dx 45° (skull)
        { objs[3], vector3D(0, 0, 13),    90.f,  .1f, color(1.0, 0.0, 0.0), MaterialType::LAMBERTIAN },                                 // down (origami_cat)
        { objs[4], vector3D(-9, 0, 9),    45.f, .05f, color(0.0, 0.0, 1.0), MaterialType::LAMBERTIAN },                                 // sx -45° (origami_dog)
        { objs[5], vector3D(-12, 0, 0),   90.f, 3.5f, color(0.5, 0.5, 0.5), MaterialType::LAMBERTIAN },                                 // sx (raptor)
        { objs[6], vector3D(-9, 0, -9),   45.f, 2.5f, color(0.5, 0.5, 0.5), MaterialType::DIELECTRIC },                                 // sx 45° (whale)
        { objs[7], vector3D(0, 0, -14),   0.f, .15f,  color(0.5, 0.5, 0.5), MaterialType::METAL }                                       // up (mandalorian)
    };

    int idx = 0;
    for (Statua s : statues) {
        build_mat_obj<<<1,1>>>(d_mats, rand_state, s.matIdx, idx, s.colore, s.obj, d_converted_mats, s.baseTextureData, s.mtlTexturesData );
        allocate_obj(d_list, s.obj, idx, rand_state, start_id, s.translate, s.rotate, s.scale, d_mats, d_converted_mats);
        start_id += s.obj.num_triangles;
        idx++;
    }
}

__device__ void create_sphere(object **d_list, curandState *rand_state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    curandState local_rand_state = *rand_state;

    // ---------------- SFERA TEST
    float radious = 0.f;
	d_list[0] = new sphere(vector3D(14.f, 2.f, 0.f), radious, new metal(color(0.95f), 0.05f));
    // ---------------- SFERA TEST

    *rand_state = local_rand_state;
}

// -- ROTAZIONE CAMERA
__global__ void build_sphere(object **d_list, curandState *rand_state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    create_sphere(d_list, rand_state);
}

void build_scene_gallery(object **d_list, objData *objs, curandState *rand_state, material **d_mat, material **d_converted_mats) {
    build_sphere<<<1,1>>>(d_list, rand_state);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    create_plane<<<1, 1>>>(d_list, 1, rand_state);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    create_statues(d_list, objs, rand_state, 2, d_mat, d_converted_mats);

    // init_world<<<1, 1>>>(d_list, d_world, num_objects);
    // CUDA_CONTROL(cudaGetLastError());
    // CUDA_CONTROL(cudaDeviceSynchronize());
}
// =================== !GALLERY ====================

// ==================== SCENA SKYBOX ====================
__global__ void build_skybox(object **d_list, curandState *rand_state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    curandState local_rand_state = *rand_state;
    float radious = 50.f;
	d_list[0] = new sphere(vector3D(0, 0, 0), radious, new metal(color(0.95f), 0.05f));
    *rand_state = local_rand_state;
}

void build_scene_skybox(object **d_list, curandState *rand_state, material **d_mat, object **d_world, int num_objects) {
    build_skybox<<<1,1>>>(d_list, rand_state);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    init_world<<<1, 1>>>(d_list, d_world, num_objects);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());
}
// =================== !SCENA SKYBOX ====================

// ==================== SCENA SFERE ====================
__device__ void create_spheres(object **d_list, curandState *rand_state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    curandState local_rand_state = *rand_state;

    int start_id = 0;
    d_list[start_id++] = new sphere(point3D(0,-1000,0), 1000,
                            new lambertianTexture(new checker_texture(new constant_texture(color(0.2, 0.3, 0.1)), new constant_texture(color(0.9, 0.9, 0.9)))));

    const int extent = 11;
    for (int a = -extent; a < extent; a++) {
        for (int b = -extent; b < extent; b++) {
            auto choose_mat = RND;
            point3D center(a + 0.9 * RND, 0.2, b + 0.9 * RND);
            if ((center - point3D(4, 0.2, 0)).length() > 0.9) {
                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color(RND * RND, RND * RND, RND * RND);
                    d_list[start_id++] = new sphere(center, 0.2, new lambertian(albedo));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color(0.5 * (1 + RND), 0.5 * (1 + RND), 0.5 * (1 + RND));
                    d_list[start_id++] = new sphere(center, 0.2, new metal(albedo, 0.5 * RND));
                } else {
                    // glass
                    d_list[start_id++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
    }

    d_list[start_id++] = new sphere(point3D(-4, 1, 0), 1.f, new emitter(new constant_texture(color(0.9))));
    d_list[start_id++] = new sphere(point3D(0, 1, 0), 1.f, new dielectric(1.5));
    d_list[start_id++] = new sphere(point3D(4, 1, 0), 1.f, new metal(color(0.7, 0.6, 0.5), 0.0));

    printf("create %d sfere\n", start_id);

    *rand_state = local_rand_state;
}

__global__ void build_spheres(object **d_list, curandState *rand_state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    create_spheres(d_list, rand_state);
}

void build_scene_spheres(object **d_list, curandState *rand_state, material **d_mat, object **d_world, int num_objects) {
    build_spheres<<<1,1>>>(d_list, rand_state);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    init_world<<<1, 1>>>(d_list, d_world, num_objects);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());
}
// =================== !SCENA SFERE ====================

// ==================== SCENA CORNEL BOX ====================
__global__ void build_cornel_box(object **d_list, curandState *rand_state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    curandState local_rand_state = *rand_state;

    int start_id = 0;

    auto red   = new lambertian(color(.65, .05, .05));
    auto white = new lambertian(color(.73, .73, .73));
    auto green = new lambertian(color(.12, .45, .15));
    auto light = new emitter(new constant_texture(color(0.8f)), 1);

    d_list[start_id++] = new yz_rect(0, 555, 0, 555, 555, green);
    d_list[start_id++] = new yz_rect(0, 555, 0, 555, 0, red);
    d_list[start_id++] = new xz_rect(213, 343, 227, 332, 554, light);
    d_list[start_id++] = new xz_rect(0, 555, 0, 555, 555, white);
    d_list[start_id++] = new xz_rect(0, 555, 0, 555, 0, white);
    d_list[start_id++] = new xy_rect(0, 555, 0, 555, 555, white);

    object *box_1 = new box2(point3D(0,0,0), point3D(165,330,165), white);
    d_list[start_id++] = new Translate(new RotateY(box_1, 15), vector3D(265,0,295));

    //object *box_2 = new box2(point3D(0,0,0), point3D(165,165,165), white);
    //d_list[start_id++] = new Translate(new RotateY(box_2, -18), vector3D(130,0,65));

    d_list[start_id++] = new sphere(point3D(130,80,300), 82.5, new metal(color(0.95f), 0.05f));

    printf("creati %d objects\n", start_id);

    *rand_state = local_rand_state;
}
__global__ void build_room(object **d_list, curandState *rand_state);

void build_scene_cornel(object **d_list, curandState *rand_state, object **d_world, int num_objects) {
    build_cornel_box<<<1,1>>>(d_list, rand_state);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    init_world<<<1, 1>>>(d_list, d_world, num_objects);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    //build_room<<<1, 1>>>(d_list, rand_state);
    //CUDA_CONTROL(cudaGetLastError());
    //CUDA_CONTROL(cudaDeviceSynchronize());
}
// =================== !SCENA CORNEL BOX ====================

// ==================== SCENA STAR WARS ====================
__global__ void create_floor_sw(object** d_list, int start_id, curandState *rand_state) {
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

    color desert(160.f/255.f, 150.f/255.f, 100.f/255.f);
    // - legno
    float density = 10.f;
    float hardness = 100.f;
    d_list[start_id] = new sphere(vector3D(0,-1000.0,-1), 1000, new lambertianTexture(new wood_texture(color(1, 193.f/255.f, 140.f/255.f), color(218.f/255.f, 109.f/255.f, 66.f/255.f), density, hardness)));
}

void create_charaters_sw(object **d_list, objData *objs, curandState *rand_state, int start_id, material **d_mats, material **d_converted_mats) {
    //color gold(224.f/255.f, 197.f/255.f, 10.f/255.f);
    color gold(1.f, 215.f/255.f ,0);
    Statua statues[] = {
        creaStatua(objs[0], vector3D(18, -0.5, -5), -45.f,   .7f, color(.95),       MaterialType::METAL),                   // (rd2d))
        creaStatua(objs[1], vector3D(14, 0, 2),     -75.f,   1.f, color(.8),        MaterialType::METAL),                   // (light saber > handle)
        creaStatua(objs[2], vector3D(14, 0, 2),     -75.f,   1.f, color(.8,0,0),    MaterialType::EMITTER),                 // (light saber > light)
        creaStatua(objs[3], vector3D(13, 0, 4),     -90.f,   1.f, color(.8),        MaterialType::METAL),                   // (light saber terrain > handle)
        creaStatua(objs[4], vector3D(13, 0, 4),     -90.f,   1.f, color(0,0,.8),    MaterialType::EMITTER),                 // (light saber terrain > light)
        creaStatua(objs[5], vector3D(11, 3, 5.8),   -110.f,  .2f, color(0.1f),      MaterialType::GLOSSY),                  // (helmet stormtrooper)
        creaStatua(objs[6], vector3D(14, 0, -1.5),  -130.f, .35f, gold,             MaterialType::METAL),                   // (c3po)
        creaStatua(objs[7], vector3D(50, -1.5, 10), -67.f,  1.4f, color(.8),        MaterialType::METAL),                   // (farming droid)
    };

    int idx = 0;
    for (Statua s : statues) {
        build_mat_obj<<<1,1>>>(d_mats, rand_state, s.matIdx, idx, s.colore, s.obj, d_converted_mats, s.baseTextureData, s.mtlTexturesData );
        allocate_obj(d_list, s.obj, idx, rand_state, start_id, s.translate, s.rotate, s.scale, d_mats, d_converted_mats);
        start_id += s.obj.num_triangles;
        idx++;
    }
}

void build_scene_star_wars(object **d_list, objData *objs, curandState *rand_state, material **d_mat, material **d_converted_mats) {
    build_sphere<<<1,1>>>(d_list, rand_state);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    create_floor_sw<<<1, 1>>>(d_list, 1, rand_state);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    create_charaters_sw(d_list, objs, rand_state, 2, d_mat, d_converted_mats);

    // init_world<<<1, 1>>>(d_list, d_world, num_objects);
    // CUDA_CONTROL(cudaGetLastError());
    // CUDA_CONTROL(cudaDeviceSynchronize());
}
// =================== !SCENA STAR WARS ====================
