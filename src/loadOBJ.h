#pragma once

#include "material.h"
#include "geometries/object.h"
#include "geometries/triangle.h"
#include "geometries/instance.h"
#include "external.h"
#include "texture.h"
#include "utils/utils.h"
#include "utils/transform.h"

struct shapeData
{
	int size;
	tinyobj::index_t* indices;
	int* material_ids;
};

struct mtlData
{
	float* diffuse_a;			// Kd
	float* specular_a;			// Ks
	float* emissive_a;			// Ke
	// - transparency;
	float* transmittance;		// 
	float dissolve;				// 
	// - sharpness_a;
	float shininess;			// 
	int illum;
	// IMAGES TEXTURE
	char* ambient_texname;             // map_Ka
	int ambient_texname_size;
	char* diffuse_texname;             // map_Kd
	int diffuse_texname_size;
	char* specular_texname;            // map_Ks
	int specular_texname_size;
	char* specular_highlight_texname;  // map_Ns
	int specular_highlight_texname_size;
	char* bump_texname;                // map_bump, map_Bump, bump
	int bump_texname_size;
};

struct objData
{
	shapeData* shapes;
	float* vertices;
	float* normals;
	int num_triangles;
	int num_shapes;
	float* textures;
	// - mtl
	mtlData* mtl;
	bool mtl_loaded = false;
	int num_materials;

	const char* basepath;
};

#define RND (curand_uniform(&local_rand_state))

__global__ void create_obj_hittables(object** d_list, int matIdx, objData obj, curandState *rand_state, int start_id = 0, float scale = 1.0f, vector3D translate = vector3D(0,0,0), float rotate = 0.0f, material **d_mat = nullptr, material **converted_mats = nullptr) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= obj.num_triangles) return;
 
	// Identify triangle ID
	int tri_count = 0;
	int tri_id = 0;
	int shape_id = 0;
	for (int s = 0; s < obj.num_shapes; s++) {
		if (idx < tri_count + obj.shapes[s].size) {
			tri_id = idx - tri_count;
			shape_id = s;
			break;
		}
		tri_count += obj.shapes[s].size;
	}

	// Triangles
	float triangle_points[9];
	float triangle_texcoord[6];
	for (int v = 0; v < 3; v++) {
		int index_offset = tri_id * 3;
		tinyobj::index_t idx = obj.shapes[shape_id].indices[index_offset + v];
		triangle_points[v*3 + 0] = obj.vertices[3*idx.vertex_index+0] * scale;
		triangle_points[v*3 + 1] = obj.vertices[3*idx.vertex_index+1] * scale;
		triangle_points[v*3 + 2] = obj.vertices[3*idx.vertex_index+2] * scale;

		// Check if `texcoord_index` is zero or positive. negative = no texcoord data
		if (idx.texcoord_index >= 0) {
			triangle_texcoord[v*2 + 0] = obj.textures[2*idx.texcoord_index+0];
			triangle_texcoord[v*2 + 1] = obj.textures[2*idx.texcoord_index+1];
		}
	}

    curandState local_rand_state = *rand_state;

	material* tri_mat;
	if (obj.mtl_loaded) {
		tri_mat = converted_mats[obj.shapes[shape_id].material_ids[0]];
	} else {
		tri_mat = d_mat[matIdx];
	}

	object* tri = new triangle(point3D(triangle_points[0], triangle_points[1], triangle_points[2]),
							   point3D(triangle_points[3], triangle_points[4], triangle_points[5]),
							   point3D(triangle_points[6], triangle_points[7], triangle_points[8]),
							   tri_mat,
							   point2D(triangle_texcoord[0], triangle_texcoord[1]),
							   point2D(triangle_texcoord[2], triangle_texcoord[3]),
							   point2D(triangle_texcoord[4], triangle_texcoord[5])
							   );
	d_list[start_id + idx] = new Translate(new RotateY(tri, rotate), translate);

    *rand_state = local_rand_state;
}

objData load_obj(const char* obj_file, const char* basepath = NULL, bool triangulate = true) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warn;
	std::string err;

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, obj_file, basepath, triangulate);

	if (!warn.empty()) {
	    std::cout << warn << std::endl;
	}

	if (!err.empty()) {
	    std::cerr << err << std::endl;
	}

	if (!ret) {
		printf("Failed to load/parse .obj\n");
	    exit(1);
	}

	objData data;
	data.num_triangles = 0;
	data.num_shapes = shapes.size();

	std::vector<shapeData> d_shapes;
	for (int i = 0; i < data.num_shapes; i++) {
		shapeData shape;
		shape.size = shapes[i].mesh.num_face_vertices.size();
		data.num_triangles += shape.size;

		cudaMalloc((tinyobj::index_t**)&shape.indices,
				   shapes[i].mesh.indices.size() * sizeof(tinyobj::index_t));
		cudaMemcpy(shape.indices, &(shapes[i].mesh.indices[0]),
				   shapes[i].mesh.indices.size() * sizeof(tinyobj::index_t),
				   cudaMemcpyHostToDevice);

		int num_material_ids = shapes[i].mesh.material_ids.size();
		cudaMalloc((int**)&shape.material_ids, num_material_ids * sizeof(int));
		cudaMemcpy(shape.material_ids, &(shapes[i].mesh.material_ids[0]), num_material_ids * sizeof(int), cudaMemcpyHostToDevice);
		
		d_shapes.push_back(shape);
	}

	printf("num shapes %i, num d_list %i\n", data.num_shapes, data.num_triangles);

	cudaMalloc((shapeData**)&data.shapes, data.num_shapes * sizeof(shapeData));
	cudaMalloc((float**)&data.vertices, attrib.vertices.size() * sizeof(float));
	cudaMalloc((float**)&data.normals, attrib.normals.size() * sizeof(float));

	cudaMemcpy(data.shapes, &(d_shapes[0]), data.num_shapes * sizeof(shapeData), cudaMemcpyHostToDevice);
	cudaMemcpy(data.vertices, &(attrib.vertices[0]), attrib.vertices.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(data.normals, &(attrib.normals[0]), attrib.normals.size() * sizeof(float), cudaMemcpyHostToDevice);

	// --- nuove proprieta'
	cudaMalloc((float**)&data.textures, attrib.texcoords.size() * sizeof(float));
	cudaMemcpy(data.textures, &(attrib.texcoords[0]), attrib.texcoords.size() * sizeof(float), cudaMemcpyHostToDevice);

	// --- mtl
	std::vector<mtlData> d_converted_mats;
	for (auto& raw_mat: materials) {
		mtlData mtl;
		mtl.dissolve = raw_mat.dissolve;
		mtl.shininess = raw_mat.shininess;
		mtl.illum = raw_mat.illum;

		cudaMalloc((float**)&mtl.diffuse_a, 3 * sizeof(float));
		cudaMemcpy(mtl.diffuse_a, &(raw_mat.diffuse[0]), 3 * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((float**)&mtl.specular_a, 3 * sizeof(float));
		cudaMemcpy(mtl.specular_a, &(raw_mat.specular[0]), 3 * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((float**)&mtl.emissive_a, 3 * sizeof(float));
		cudaMemcpy(mtl.emissive_a, &(raw_mat.emission[0]), 3 * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((float**)&mtl.transmittance, 3 * sizeof(float));
		cudaMemcpy(mtl.transmittance, &(raw_mat.transmittance[0]), 3 * sizeof(float), cudaMemcpyHostToDevice);

		// IMAGES TEXTURE
		std::string ambient_texname = raw_mat.ambient_texname;
		std::vector<char> ambient_texname_c(ambient_texname.c_str(), ambient_texname.c_str() + ambient_texname.size() + 1);
		cudaMalloc((char**)&mtl.ambient_texname, ambient_texname_c.size() * sizeof(char));
		cudaMemcpy(mtl.ambient_texname, &(ambient_texname_c[0]), ambient_texname_c.size() * sizeof(char), cudaMemcpyHostToDevice);
		mtl.ambient_texname_size = ambient_texname_c.size() * sizeof(char);

		std::string diffuse_texname = raw_mat.diffuse_texname;
		std::vector<char> diffuse_texname_c(diffuse_texname.c_str(), diffuse_texname.c_str() + diffuse_texname.size() + 1);
		cudaMalloc((char**)&mtl.diffuse_texname, diffuse_texname_c.size() * sizeof(char));
		cudaMemcpy(mtl.diffuse_texname, &(diffuse_texname_c[0]), diffuse_texname_c.size() * sizeof(char), cudaMemcpyHostToDevice);
		mtl.diffuse_texname_size = diffuse_texname_c.size() * sizeof(char);

		std::string specular_texname = raw_mat.specular_texname;
		std::vector<char> specular_texname_c(specular_texname.c_str(), specular_texname.c_str() + specular_texname.size() + 1);
		cudaMalloc((char**)&mtl.specular_texname, specular_texname_c.size() * sizeof(char));
		cudaMemcpy(mtl.specular_texname, &(specular_texname_c[0]), specular_texname_c.size() * sizeof(char), cudaMemcpyHostToDevice);
		mtl.specular_texname_size = specular_texname_c.size() * sizeof(char);

		std::string specular_highlight_texname = raw_mat.specular_highlight_texname;
		std::vector<char> specular_highlight_texname_c(specular_highlight_texname.c_str(), specular_highlight_texname.c_str() + specular_highlight_texname.size() + 1);
		cudaMalloc((char**)&mtl.specular_highlight_texname, specular_highlight_texname_c.size() * sizeof(char));
		cudaMemcpy(mtl.specular_highlight_texname, &(specular_highlight_texname_c[0]), specular_highlight_texname_c.size() * sizeof(char), cudaMemcpyHostToDevice);
		mtl.specular_highlight_texname_size = specular_highlight_texname_c.size() * sizeof(char);

		std::string bump_texname = raw_mat.bump_texname;
		std::vector<char> bump_texname_c(bump_texname.c_str(), bump_texname.c_str() + bump_texname.size() + 1);
		cudaMalloc((char**)&mtl.bump_texname, bump_texname_c.size() * sizeof(char));
		cudaMemcpy(mtl.bump_texname, &(bump_texname_c[0]), bump_texname_c.size() * sizeof(char), cudaMemcpyHostToDevice);
		mtl.bump_texname_size = bump_texname_c.size() * sizeof(char);

		data.basepath = basepath;

		d_converted_mats.push_back(mtl);
	}
	data.num_materials = materials.size();
	if (data.num_materials > 0) data.mtl_loaded = true;
	cudaMalloc((mtlData**)&data.mtl, data.num_materials * sizeof(mtlData));
	cudaMemcpy(data.mtl, &(d_converted_mats[0]), data.num_materials * sizeof(mtlData), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
    cudaError cudaErr = cudaGetLastError();
    if (cudaSuccess != cudaErr) {
        fprintf(stderr, "cudaCheckError() failed at copying obj to device: %s\n", cudaGetErrorString(cudaErr));
        exit(-1);
    }

	return data;
}
