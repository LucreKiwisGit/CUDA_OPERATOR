#include "utils.h"
#include <iostream>
#include <getopt.h>
#include <cstring>

void printHelpInfo() {

    std::cout << "Usage: ./main [--help] [--op_name] [--op_type] [--input_file] [--output_file]" << std::endl;
    std::cout << "--help: print help information" << std::endl;
    std::cout << "--op_name: operator name, e.g. Conv2d, GEMM, ImplicitGEMM" << std::endl;
    std::cout << "--op_type: operator type, e.g. FLOAT16, FLOAT32, INT32, INT8" << std::endl;
 
}

static struct option long_options[] = {
    {"help", no_argument, 0, 'h'},
    {"op_name", required_argument, 0, 'n'},
    {"op_type", required_argument, 0, 't'},
    {0, 0, 0, 0}
};

bool SampleTest::init_args(int argc, char** argv) {
    if (argc == 1) {
        printHelpInfo();
        return false;
    }

    int long_index = 0;
    int c;

    while ((c = getopt_long(argc, argv, "hn:t:", long_options, &long_index)) != -1) {
        switch (c) {
            case 'h':
                printHelpInfo();
                return false;
            case 'n':
                if (strcmp(optarg, "Conv2d") == 0) {
                    args.op_name = Conv2d;
                } else if (strcmp(optarg, "GEMM") == 0) {
                    args.op_name = GEMM;
                } else if (strcmp(optarg, "ImplicitGEMM") == 0) {
                    args.op_name = ImplicitGEMM;
                }
                else if (strcmp(optarg, "Im2colGEMM") == 0) {
                    args.op_name = Im2colGEMM;
                }
                else {
                    printHelpInfo();
                    return false;
                }
                break;
            case 't':
                if (strcmp(optarg, "FLOAT16") == 0) {
                    args.op_type = FLOAT16;
                } else if (strcmp(optarg, "FLOAT32") == 0) {
                    args.op_type = FLOAT32;
                } else if (strcmp(optarg, "INT32") == 0) {
                    args.op_type = INT32;
                }
                else {
                    printHelpInfo();
                    return false;
                }
                break;
            default:
                printHelpInfo();
                return false;
        }
    }

    return true;
}

void SampleTest::run_test()
{
    if (args.op_name == Operator::Conv2d) {
        std::cout << "Executing GEMM test with data type: " << args.op_type << "\n";
        // gemm_test(args.op_type);
    }
    else if (args.op_name == Operator::GEMM) {
        std::cout << "Executing GEMM test with data type: " << args.op_type << "\n";
        // gemm_test(args.op_type);
        gemm_fp32_test();
    }
    else if (args.op_name == Operator::ImplicitGEMM) {
        std::cout << "Executing Implicit GEMM test with data type: " << args.op_type << "\n";
        implicit_gemm_fp16_test();
    }
    else if (args.op_name == Operator::Im2colGEMM) {
        std::cout << "Executing Im2col GEMM test with data type: " << args.op_type << "\n";
        im2col_gemm_fp16_test();
    }
    else {
        std::cout << "Unsupported operator type: " << args.op_name << "\n";
    }
}