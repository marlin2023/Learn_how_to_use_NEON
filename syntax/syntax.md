# NEON指令

本部分对常用的NEON指令进行介绍，会先介绍NEON intrinsics函数 ，然后再介绍与之对应的NEON汇编指令，最后跟着简单的例子。（实际开发中，即便是写汇编代码，使用intrinsics也有好处。先用intrinsics写好代码编译后在反汇编，在此基础上进行优化，可能比较省力。）

NEON intrinsics 中的寄存器的分配工作是编译器来做，让你把精力专注在算法实现上。在代码的维护上，也比使用汇编实现维护起来更简单。 NEON intrinsics函数定义在arm_neon.h，该头文件中也定义一系列 vector types.

