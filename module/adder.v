// 文件名: adder.v
// 4位加法器模块

module adder (
    input [3:0] a,        // 输入A
    input [3:0] b,        // 输入B
    output [4:0] sum      // 结果 (4位和 + 1位进位)
);

    assign sum = a + b;

endmodule // adder
