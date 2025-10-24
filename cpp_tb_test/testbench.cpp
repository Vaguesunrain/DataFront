#include <iostream>
#include "Vcounter.h"       // 包含Verilator生成的我们设计的头文件
#include "verilated.h"      // 包含Verilator的核心库
#include "verilated_vcd_c.h" 

int main(int argc, char** argv) {
    // 初始化Verilator
    Verilated::commandArgs(argc, argv);

    // 实例化我们的Verilog模块
    Vcounter* top = new Vcounter;

    // 初始化波形记录
    Verilated::traceEverOn(true);
    VerilatedVcdC* m_trace = new VerilatedVcdC;
    top->trace(m_trace, 99); // 99是追踪深度
    m_trace->open("waveform.vcd");

    // 仿真主循环
    int sim_time = 0;
    while (sim_time < 200) {
        // --- 时钟下降沿 ---
        top->clk = 0;
        top->eval(); // 评估模型
        m_trace->dump(sim_time * 10 + 5); // 在特定时间点记录波形

        // --- 时钟上升沿 ---
        top->clk = 1;

        // 在时钟上升沿前，根据仿真时间改变输入信号
        if (sim_time < 2) {
            top->rst = 1; // 初始时复位
        } else {
            top->rst = 0;
        }

        top->eval(); // 再次评估模型，捕获上升沿的变化
        m_trace->dump(sim_time * 10 + 10); // 记录波形

        // 打印输出信号，进行检查
        std::cout << "Time: " << sim_time
                  << " RST: " << (int)top->rst
                //   << " EN: " << (int)top->en
                  << " Count: " << (int)top->result << std::endl;

        // 仿真时间前进
        sim_time++;
    }

    // 清理
    m_trace->close();
    delete top;
    return 0;
}