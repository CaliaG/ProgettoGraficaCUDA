#pragma once

#include "../loadOBJ.h"

struct ImageData {
    float* d_imgData = nullptr;
    int width = 0;
    int height = 0;
};

struct MTLTextureData {
	ImageData ambientTex;
	ImageData diffuseTex;
	ImageData specularTex;
	ImageData specularHighlightTex;
	ImageData bumpTex;
};

ImageData caricaImmagine(char imagePath[] = nullptr) {
    int w, h, ch;
    //stbi_ldr_to_hdr_scale(1.0f);
    //stbi_ldr_to_hdr_gamma(1.0f);
    float* imgData_h = stbi_loadf(imagePath, &w, &h, &ch, 0);
    std::cout << "Loaded image with " << w << "x" << h << " and " << ch << " channels\n";
    float* d_imgData;
    size_t imgSize = w * h * ch * sizeof(float);
    CUDA_CONTROL(cudaMalloc((float**)&d_imgData, imgSize));
    CUDA_CONTROL(cudaMemcpy(d_imgData, imgData_h, imgSize, cudaMemcpyHostToDevice));
    stbi_image_free(imgData_h);
    return { d_imgData, w, h };
}

char* getImagePath(char* texname, int size, const char* basepath) {
    if (size < 4) return nullptr; // deve avere almeno un carattere in piu' rispetto all'estensione (_.png/_.jpg)

    char* h_textname = (char*)malloc(size * sizeof(char));
    cudaMemcpy(h_textname, texname, size * sizeof(char), cudaMemcpyDeviceToHost);
    std::string imagePathStr = std::string(basepath) + h_textname;
    char* imagePath = (char*)malloc(strlen(imagePathStr.c_str()) + 1);
    strcpy(imagePath, imagePathStr.c_str());
    //printf("• texname: %s [%d] > path: %s\n", h_textname, size, imagePath);
    return (char*)imagePath;
}

MTLTextureData* caricaTextureMTL(objData obj) {
    printf("============================================================\n");
    printf(" CARICA TEXTURES MTL\n");
    printf("============================================================\n");
    printf(" BASEPATH: %s - mtl size: %d\n", obj.basepath, obj.num_materials);

    std::vector<MTLTextureData> d_mtlTexturesData;
    MTLTextureData* mtlTexturesData;

    mtlData* mtl = (mtlData*)malloc(obj.num_materials * sizeof(mtlData));
    CUDA_CONTROL(cudaMemcpy(mtl, obj.mtl, obj.num_materials * sizeof(mtlData), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < obj.num_materials; i++) {
        printf("\n MTL N°: %d\n", i);
        MTLTextureData mtlTexs;
        // ambient
        char* ambientPath = getImagePath(mtl[i].ambient_texname, mtl[i].ambient_texname_size, obj.basepath);
        printf(" • ambient path: %s\n", ambientPath);
        if (ambientPath != nullptr) {
            ImageData ambientData = caricaImmagine(ambientPath);
            mtlTexs.ambientTex = ambientData;
        }
        // diffuse
        char* diffusePath = getImagePath(mtl[i].diffuse_texname, mtl[i].diffuse_texname_size, obj.basepath);
        printf(" • diffuse path: %s\n", diffusePath);
        if (diffusePath != nullptr) {
            ImageData diffuseData = caricaImmagine(diffusePath);
            mtlTexs.diffuseTex = diffuseData;
        }
        // specular
        char* specularPath = getImagePath(mtl[i].specular_texname, mtl[i].specular_texname_size, obj.basepath);
        printf(" • specular path: %s\n", specularPath);
        if (specularPath != nullptr) {
            ImageData specularTex = caricaImmagine(specularPath);
            mtlTexs.specularTex = specularTex;
        }
        // specular_highlight
        char* specularHighlightPath = getImagePath(mtl[i].specular_highlight_texname, mtl[i].specular_highlight_texname_size, obj.basepath);
        printf(" • specular_highlight path: %s\n", specularHighlightPath);
        if (specularHighlightPath != nullptr) {
            ImageData specularHighlightTex = caricaImmagine(specularHighlightPath);
            mtlTexs.specularHighlightTex = specularHighlightTex;
        }
        // bump
        char* bumpPath = getImagePath(mtl[i].bump_texname, mtl[i].bump_texname_size, obj.basepath);
        printf(" • bump path: %s\n", bumpPath);
        if (bumpPath != nullptr) {
            ImageData bumpTex = caricaImmagine(bumpPath);
            mtlTexs.bumpTex = bumpTex;
        }

        d_mtlTexturesData.push_back(mtlTexs);
    }
    CUDA_CONTROL(cudaMalloc((MTLTextureData**)&mtlTexturesData, obj.num_materials * sizeof(MTLTextureData)));
    CUDA_CONTROL(cudaMemcpy(mtlTexturesData, &(d_mtlTexturesData[0]), obj.num_materials * sizeof(MTLTextureData), cudaMemcpyHostToDevice));

    printf("============================================================\n");
    return mtlTexturesData;
}
