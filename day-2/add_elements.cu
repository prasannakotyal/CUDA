#include <iostream>
#include <math.h>
using namespace std;

__global__
void add(int n, float* x,float *y){
    
    int index = blockIdx.x*blockDim.x+ threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    for(int i=index;i<n;i+=stride){
        y[i]+=x[i];
    }
}

int main(){

    int N = 1<<20;
    float *x,*y;
    cudaMallocManaged(&x,N*sizeof(float));
    cudaMallocManaged(&y,N*sizeof(float));

    for(int i=0;i<N;i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (N+blockSize-1)/(blockSize);

    cudaMemLocation loc;
    loc.type = cudaMemLocationTypeDevice;
    loc.id = 0;  // device 0
    cudaMemPrefetchAsync(x, N * sizeof(float), loc, 0, 0);
    cudaMemPrefetchAsync(y,N*sizeof(float),loc,0,0);
    add<<<numBlocks,blockSize>>>(N,x,y);
    cudaDeviceSynchronize();
    float max_error = 0.0f;
    for(int i=0;i<N;i++){
        max_error = fmax(max_error,fabs(y[i]-3.0f));
    }

    cout<<"Max error: "<<max_error<<endl;

    cudaFree(x);
    cudaFree(y);

    return 0;
}