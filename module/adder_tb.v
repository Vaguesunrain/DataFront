// 文件名: adder_tb.v
// 4位加法器测试平台

`timescale 1ns/1ps

module adder_tb;

    // 声明待测试模块的信号
    reg [3:0] a;
    reg [3:0] b;
    wire [4:0] sum;

    // 实例化待测试模块
    adder uut (
        .a(a),
        .b(b),
        .sum(sum)
    );

    // 文件输出设置 (用于GTKWave)
    initial begin
        // 指定输出的波形文件名为 waves.vcd
        $dumpfile("waves.vcd");
        // 指定要记录的信号 (这里记录所有信号)
        $dumpvars(0, adder_tb);
    end

    // 时钟和激励
    initial begin
        // 初始值
        a = 4'b0000;
        b = 4'b0000;

        // 激励1
        #10 a = 4'b0101; // 5
        b = 4'b0011; // 3
        
        // 激励2
        #10 a = 4'b1111; // 15
        b = 4'b0001; // 1
        
        // 激励3 (产生进位)
        #10 a = 4'b1000; // 8
        b = 4'b1000; // 8
        
        // 结束仿真
        #10 $finish;
    end

endmodule // adder_tb
