#include "caffe/util/math_functions.hpp"

namespace caffe {

int param_len = 0;
int **a;
int **a_host;
bool **b;
bool **b_host;

template <typename Dtype>
__global__ void SGDUpdate(int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate, float dvf_threshold, int* per_block_results, bool* whether_update) {
  __shared__ int sdata[CAFFE_CUDA_NUM_THREADS];
  sdata[threadIdx.x] = 0;
  __syncthreads();
  CUDA_KERNEL_LOOP(i, N) {
    // threshold works here
    // g[i] denotes the updates computed in this iteration
    // we first aggregated g[i] with h[i]
    // h[i] is the update computed in last iteration
    g[i] = h[i] = momentum*h[i] + local_rate*g[i];
    // ***********************************************************
    // How to offload following function to SmartNIC
    // MPI cannot be used anymore if we offload following function to SmartNIC
    // ***********************************************************
    if (g[i] >= 0 && g[i] <= dvf_threshold) {
      // here we drop g[i], store the dropped value in h[i]
      // this is different with the paper, since we use momentum SGD
      g[i] = 0;
      h[i] = 1.0/momentum * h[i]; // this is a trick for h[i], must use it
      // sdata is used to trace which update is useful
      sdata[threadIdx.x] += 1;
      whether_update[i] = false;
    } else if (g[i] < 0 && g[i] >= -dvf_threshold) {
      // here we dlso rop g[i], store the dropped value in h[i]
      g[i] = 0;
      h[i] = 1.0/momentum * h[i];
      sdata[threadIdx.x] += 1;
      whether_update[i] = false;
    } else {
      // here we do not drop g[i], store the dropped value in h[i]
      whether_update[i] = true;
    }
  }
  __syncthreads();
  for(int offset = blockDim.x/2; offset > 0; offset >>= 1) {  
    if(threadIdx.x < offset) {  
      // sdata is used to trace which update is useful
       sdata[threadIdx.x] += sdata[threadIdx.x + offset];  
    }  
    __syncthreads();  
  }  
  if(threadIdx.x == 0) {  
    per_block_results[blockIdx.x] = sdata[0];  
  }
}

template <typename Dtype>
long sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate, float dvf_threshold, int* update_vector, int param_id) {
  long dn = 0;
  size_t block_num = CAFFE_GET_BLOCKS(N);
  int *d_partial_sums = 0;
  int *d_partial_sums_host = 0;
  bool *whether_update = 0;
  bool *whether_update_host = 0;
  if (param_id == param_len) {
    param_len += 1;
    cudaMalloc((void**)&d_partial_sums, sizeof(int) * block_num);
    d_partial_sums_host = (int*) malloc(sizeof(int) * block_num);
    cudaMalloc((void**)&whether_update, sizeof(bool) * N);
    whether_update_host = (bool*) malloc(sizeof(bool) * N);

    int **a_tmp = (int**)malloc(sizeof(int*) * param_len);
    int **a_host_tmp = (int**)malloc(sizeof(int*) * param_len);
    bool **b_tmp = (bool**)malloc(sizeof(bool*) * param_len);
    bool **b_host_tmp = (bool**)malloc(sizeof(bool*) * param_len);

    for (int i=0; i<(param_len-1); i++) {
      a_tmp[i] = a[i];
      a_host_tmp[i] = a_host[i];
      b_tmp[i] = b[i];
      b_host_tmp[i] = b_host[i];
    }
    a_tmp[param_len-1] = d_partial_sums;
    a_host_tmp[param_len-1] = d_partial_sums_host;
    b_tmp[param_len-1] = whether_update;
    b_host_tmp[param_len-1] = whether_update_host;
    free(a);
    free(a_host);
    free(b);
    free(b_host);
    a = a_tmp;
    a_host = a_host_tmp;
    b = b_tmp;
    b_host = b_host_tmp;
  } else {
    d_partial_sums = a[param_id];
    d_partial_sums_host = a_host[param_id];
    whether_update = b[param_id];
    whether_update_host = b_host[param_id];
  }

  SGDUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<block_num, CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, momentum, local_rate, dvf_threshold, d_partial_sums, whether_update);
  CUDA_POST_KERNEL_CHECK;

  //**************************************************************
  // Communication happends here, delete the MPI related communication function
  // need to implemente the communication  function using smart NIC APIs
  // following is what MPI do in previous implementation:
  //    MPI_Send g to rank 0, in Rank 0, g[i] += receviced_g[i]
  //    Then, in Rank0, average the update: g[i] = g[i] / RankNumber
  //    MPI_Bcast g to other ranks
  // g would be used to update parameteres in another function, and we would not care it.
  //*************************************************************

  cudaMemcpy(d_partial_sums_host, d_partial_sums, sizeof(int)*block_num, cudaMemcpyDeviceToHost);
  cudaMemcpy(whether_update_host, whether_update, sizeof(bool)*N, cudaMemcpyDeviceToHost);

  for (int i=0; i < block_num; i++) {
    dn += d_partial_sums_host[i];
  } 
  for (int i=0; i < N; i++) {
    update_vector[i] += whether_update_host[i];
  }

  // cudaFree(d_partial_sums);
  // cudaFree(whether_update);
  // free(d_partial_sums_host);
  // free(whether_update_host);
  return dn;
}
template long sgd_update_gpu<float>(int, float*, float*, float, float, float, int*, int);
template long sgd_update_gpu<double>(int, double*, double*, double, double, float, int*, int);

}  // namespace caffe
