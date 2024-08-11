#ifndef VERIFY_H
#define VERIFY_H

#include <cmath>
#include "convs2d.h"


// 获取给定浮点数的精度值
inline float getPrecision(float tmp)
{
    int tmpInt = (int)tmp;
    float eNum = 1.0e-6;
    if(abs(tmpInt) > 0)
    {
        while(tmpInt != 0)
        {
            tmpInt = (int)(tmpInt / 10);
            eNum *= 10;
        }
    }
    else
    {
        
        if(tmp == 0)
            return eNum;
            
        eNum = 1.0e-5;
        
        while(tmpInt == 0)
        {
            tmp *= 10;
            tmpInt = (int)(tmp);
            eNum /= 10;
        }
    }

    return eNum;
}

// CPU原始实现, 用于对比精度
void direct_conv2dCpu(param_t param);

#endif