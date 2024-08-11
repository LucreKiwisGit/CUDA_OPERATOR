#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>

#include "operators.h"

// 数据枚举类型
enum DataType{
    FLOAT32,
    FLOAT16,
    INT8,
    INT32
};

enum Operator{
    GEMM,
    ImplicitGEMM,
    Conv2d
};

typedef struct args_t
{
    Operator op_name;    // 算子名称
    DataType op_type;    // 数据类型
} args_t;


class SampleTest {
    public:
        SampleTest() {}

        bool init_args(int argc, char** argv);

        void print_helpInfo();

        void run_test();
    private:
        args_t args;
};

// bool SampleTest::init_args(int argc, char** argv);
// void SampleTest::print_helpInfo();



#endif