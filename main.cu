#include <stdio.h>
#include <cuda_runtime.h>
// #include "verify.h"
#include "convs2d.h"
#include "utils.h"

int main(int argc, char **argv) {
    SampleTest sample;
    sample.init_args(argc, argv);
    sample.run_test();
}